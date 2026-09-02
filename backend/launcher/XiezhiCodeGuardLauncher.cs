using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Drawing;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Windows.Forms;

namespace XiezhiCodeGuardLauncher
{
    internal static class Program
    {
        private const string AppTitle = "獬豸安码";
        private const int DefaultStartupTimeoutSeconds = 90;
        private const int FrontendCloseDebounceMilliseconds = 1000;
        private const int MonitorPollMilliseconds = 250;
        private static string lastBackendStdoutLog = "";
        private static string lastBackendStderrLog = "";

        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            string appDir = AppDomain.CurrentDomain.BaseDirectory;
            string backendDir = Path.Combine(appDir, "backend");
            string appPy = Path.Combine(backendDir, "app.py");
            if (!File.Exists(appPy))
            {
                ShowTopMostMessage("backend\\app.py was not found. Please run this launcher from the project root directory.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            string assetsDir = Path.Combine(backendDir, "launcher", "assets");
            string loadingImage = Path.Combine(assetsDir, "loading_file.png");
            string iconPath = Path.Combine(assetsDir, "app.ico");

            SplashForm splash = new SplashForm(loadingImage, iconPath);
            splash.Show();
            splash.UpdateStatus("Checking runtime", 5, 25);
            Application.DoEvents();

            int port = 3000;
            Process backend = null;

            try
            {
                splash.UpdateStatus("Checking fixed port 3000", 12, GetStartupTimeoutSeconds());
                Application.DoEvents();
                if (!IsPortAvailable(port))
                {
                    splash.Close();
                    ShowTopMostMessage(
                        "Port 3000 is already in use.\n\nPlease close the old Xiezhi CodeGuard process or stop the program that is using port 3000, then launch again.",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning
                    );
                    return;
                }
                backend = StartBackend(backendDir, port);
                splash.UpdateStatus("Starting backend service", 28, GetStartupTimeoutSeconds());
                Application.DoEvents();
                string url = "http://127.0.0.1:" + port + "/";
                int startupTimeout = GetStartupTimeoutSeconds();
                if (!WaitForServer(url, startupTimeout, splash, backend))
                {
                    splash.Close();
                    string detail = backend != null && backend.HasExited
                        ? "The Python process exited before the backend became ready."
                        : "The backend did not answer within " + startupTimeout + " seconds.";
                    if (!string.IsNullOrEmpty(lastBackendStderrLog))
                    {
                        detail += "\n\nBackend error log:\n" + lastBackendStderrLog;
                    }
                    ShowTopMostMessage(
                        detail + "\n\nCheck Python/dependencies and retry.",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Warning
                    );
                    StopBackend(backend);
                    return;
                }

                splash.UpdateStatus("Opening frontend browser", 96, 1);
                Application.DoEvents();
                StartBrowser(url);
                splash.UpdateStatus("Ready", 100, 0);
                Application.DoEvents();
                Thread.Sleep(350);
                splash.Close();

                MonitorFrontend(url, backend);
            }
            catch (Exception ex)
            {
                if (splash != null && !splash.IsDisposed)
                {
                    splash.Close();
                }
                ShowTopMostMessage(ex.Message, MessageBoxButtons.OK, MessageBoxIcon.Error);
                StopBackend(backend);
            }
        }

        private static Process StartBackend(string backendDir, int port)
        {
            string python = ResolvePythonExecutable(backendDir);
            string pythonArguments = "-c \"from app import create_app; app=create_app(); app.run(host='127.0.0.1', port=" + port + ", debug=False, use_reloader=False, threaded=True)\"";
            if (string.Equals(Path.GetFileName(python), "py.exe", StringComparison.OrdinalIgnoreCase))
            {
                pythonArguments = "-3.12 " + pythonArguments;
            }

            string logDir = Path.Combine(backendDir, "logs");
            Directory.CreateDirectory(logDir);
            string stamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
            lastBackendStdoutLog = Path.Combine(logDir, "launcher-backend-" + stamp + ".stdout.log");
            lastBackendStderrLog = Path.Combine(logDir, "launcher-backend-" + stamp + ".stderr.log");

            ProcessStartInfo psi = new ProcessStartInfo();
            psi.FileName = python;
            psi.Arguments = pythonArguments;
            psi.WorkingDirectory = backendDir;
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            psi.EnvironmentVariables["XIEZHI_PORT"] = port.ToString();
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            Process process = Process.Start(psi);
            if (process == null)
            {
                throw new InvalidOperationException("Unable to start the Python backend.");
            }
            StreamWriter stdout = new StreamWriter(lastBackendStdoutLog, false);
            StreamWriter stderr = new StreamWriter(lastBackendStderrLog, false);
            stdout.AutoFlush = true;
            stderr.AutoFlush = true;
            stdout.WriteLine("Launcher Python: " + python);
            process.OutputDataReceived += (sender, args) =>
            {
                if (args.Data != null)
                {
                    lock (stdout) { stdout.WriteLine(args.Data); }
                }
            };
            process.ErrorDataReceived += (sender, args) =>
            {
                if (args.Data != null)
                {
                    lock (stderr) { stderr.WriteLine(args.Data); }
                }
            };
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            return process;
        }

        private static string ResolvePythonExecutable(string backendDir)
        {
            string configured = Environment.GetEnvironmentVariable("XIEZHI_PYTHON");
            if (!string.IsNullOrWhiteSpace(configured))
            {
                string explicitPython = configured.Trim().Trim('"');
                if (File.Exists(explicitPython))
                {
                    return explicitPython;
                }
            }

            List<string> candidates = new List<string>();
            string appDir = AppDomain.CurrentDomain.BaseDirectory;
            candidates.Add(Path.Combine(appDir, "python", "python.exe"));
            candidates.Add(Path.Combine(appDir, ".venv", "Scripts", "python.exe"));
            candidates.Add(Path.Combine(appDir, "venv", "Scripts", "python.exe"));
            candidates.Add(Path.Combine(backendDir, ".venv", "Scripts", "python.exe"));
            candidates.Add(Path.Combine(backendDir, "venv", "Scripts", "python.exe"));

            string pathPython = ResolveOnPath("python.exe");
            if (!string.IsNullOrWhiteSpace(pathPython))
            {
                candidates.Add(pathPython);
            }

            string condaPrefix = Environment.GetEnvironmentVariable("CONDA_PREFIX");
            if (!string.IsNullOrWhiteSpace(condaPrefix))
            {
                candidates.Add(Path.Combine(condaPrefix, "python.exe"));
            }
            string driveRoot = Path.GetPathRoot(appDir);
            if (!string.IsNullOrWhiteSpace(driveRoot))
            {
                candidates.Add(Path.Combine(driveRoot, "software", "Anaconda3", "python.exe"));
            }
            candidates.Add(@"C:\ProgramData\Anaconda3\python.exe");

            foreach (string candidate in candidates.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (File.Exists(candidate) && SupportsModelRuntime(candidate))
                {
                    return candidate;
                }
            }

            // Keep the UI startable even on an incomplete installation so it
            // can display the concrete missing-dependency reason.
            foreach (string candidate in candidates)
            {
                if (!string.IsNullOrWhiteSpace(candidate) && File.Exists(candidate))
                {
                    return candidate;
                }
            }

            string windowsDir = Environment.GetFolderPath(Environment.SpecialFolder.Windows);
            string pyLauncher = Path.Combine(windowsDir, "py.exe");
            if (File.Exists(pyLauncher))
            {
                return pyLauncher;
            }

            return "python.exe";
        }

        private static bool SupportsModelRuntime(string python)
        {
            const string script =
                "import importlib.util,sys;" +
                "mods=('flask','joblib','sklearn','xgboost','torch','torch_geometric','transformers','safetensors');" +
                "sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in mods) else 1)";
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = python;
                psi.Arguments = "-c \"" + script + "\"";
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardError = true;
                using (Process process = Process.Start(psi))
                {
                    if (process == null)
                    {
                        return false;
                    }
                    if (!process.WaitForExit(15000))
                    {
                        process.Kill();
                        return false;
                    }
                    return process.ExitCode == 0;
                }
            }
            catch
            {
                return false;
            }
        }

        private static string ResolveOnPath(string executable)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = "where.exe";
                psi.Arguments = executable;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardError = true;
                using (Process process = Process.Start(psi))
                {
                    if (process == null)
                    {
                        return "";
                    }
                    string output = process.StandardOutput.ReadToEnd();
                    process.WaitForExit(3000);
                    foreach (string line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                    {
                        string candidate = line.Trim();
                        if (File.Exists(candidate))
                        {
                            return candidate;
                        }
                    }
                }
            }
            catch
            {
                return "";
            }
            return "";
        }

        private static int GetStartupTimeoutSeconds()
        {
            int value;
            string raw = Environment.GetEnvironmentVariable("XIEZHI_STARTUP_TIMEOUT_SECONDS");
            if (int.TryParse(raw, out value))
            {
                return Math.Max(30, Math.Min(300, value));
            }
            return DefaultStartupTimeoutSeconds;
        }

        private static void StartBrowser(string url)
        {
            ProcessStartInfo browserInfo = new ProcessStartInfo();
            browserInfo.FileName = url;
            browserInfo.UseShellExecute = true;
            Process.Start(browserInfo);
        }

        private static void MonitorFrontend(string baseUrl, Process backend)
        {
            Thread.Sleep(500);
            string statusUrl = baseUrl.TrimEnd('/') + "/api/launcher/status";
            bool hasSeenFrontend = false;
            DateTime? missingSince = null;
            bool promptedForCurrentClosure = false;
            while (true)
            {
                if (backend == null || backend.HasExited)
                {
                    return;
                }
                try
                {
                    HttpWebRequest request = (HttpWebRequest)WebRequest.Create(statusUrl);
                    request.Timeout = 800;
                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                    using (StreamReader reader = new StreamReader(response.GetResponseStream()))
                    {
                        string body = reader.ReadToEnd().ToLowerInvariant();
                        bool active = JsonBool(body, "active");
                        bool everConnected = JsonBool(body, "ever_connected");
                        bool frontendClosed = JsonBool(body, "frontend_closed");
                        bool shutdownRequested = JsonBool(body, "shutdown_requested");
                        if (shutdownRequested)
                        {
                            StopBackend(backend);
                            return;
                        }
                        if (active)
                        {
                            hasSeenFrontend = true;
                            missingSince = null;
                            promptedForCurrentClosure = false;
                        }
                        else if (frontendClosed || hasSeenFrontend || everConnected)
                        {
                            if (!promptedForCurrentClosure && !missingSince.HasValue)
                            {
                                missingSince = DateTime.Now;
                            }
                            else if (!promptedForCurrentClosure &&
                                missingSince.HasValue &&
                                (DateTime.Now - missingSince.Value).TotalMilliseconds >= FrontendCloseDebounceMilliseconds)
                            {
                                DialogResult choice = ConfirmStopServices();
                                promptedForCurrentClosure = true;
                                if (choice == DialogResult.Yes)
                                {
                                    StopBackend(backend);
                                    return;
                                }
                                missingSince = null;
                            }
                        }
                    }
                }
                catch
                {
                    Thread.Sleep(500);
                }
                Thread.Sleep(MonitorPollMilliseconds);
            }
        }

        private static DialogResult ConfirmStopServices()
        {
            return ShowTopMostMessage(
                "检测页面已关闭。\n\n是否同时停止本项目服务？",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question
            );
        }

        private static bool JsonBool(string body, string key)
        {
            string compactNeedle = "\"" + key + "\":true";
            string spacedNeedle = "\"" + key + "\": true";
            return body.Contains(compactNeedle) || body.Contains(spacedNeedle);
        }

        private static DialogResult ShowTopMostMessage(string text, MessageBoxButtons buttons, MessageBoxIcon icon)
        {
            using (Form owner = new Form())
            {
                owner.StartPosition = FormStartPosition.Manual;
                owner.Size = new Size(1, 1);
                owner.Location = new Point(-2000, -2000);
                owner.ShowInTaskbar = false;
                owner.TopMost = true;
                owner.Show();
                owner.Activate();
                return MessageBox.Show(owner, text, AppTitle, buttons, icon);
            }
        }

        private static int FindFreePort()
        {
            TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            int port = ((IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();
            return port;
        }

        private static bool IsPortAvailable(int port)
        {
            TcpListener listener = null;
            try
            {
                listener = new TcpListener(IPAddress.Loopback, port);
                listener.Start();
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                if (listener != null)
                {
                    listener.Stop();
                }
            }
        }

        private static bool WaitForServer(string url, int seconds, SplashForm splash, Process backend)
        {
            for (int i = 0; i < seconds * 2; i++)
            {
                if (backend != null && backend.HasExited)
                {
                    return false;
                }
                int elapsedHalfSeconds = i;
                int remaining = Math.Max(0, seconds - elapsedHalfSeconds / 2);
                int progress = Math.Min(95, 35 + (int)(60.0 * i / Math.Max(1, seconds * 2)));
                splash.UpdateStatus("Waiting for backend", progress, remaining);
                Application.DoEvents();
                try
                {
                    HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
                    request.Timeout = 800;
                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                    {
                        return true;
                    }
                }
                catch
                {
                    Thread.Sleep(500);
                }
            }
            return false;
        }

        private static void StopBackend(Process backend)
        {
            if (backend == null || backend.HasExited)
            {
                return;
            }
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = "taskkill";
                psi.Arguments = "/PID " + backend.Id + " /T /F";
                psi.CreateNoWindow = true;
                psi.UseShellExecute = false;
                Process killer = Process.Start(psi);
                if (killer != null)
                {
                    killer.WaitForExit(5000);
                }
            }
            catch
            {
                try { backend.Kill(); } catch { }
            }
        }

    }

    internal class SplashForm : Form
    {
        private readonly ProgressBar progressBar;

        public SplashForm(string backgroundPath, string iconPath)
        {
            Text = "Starting Xiezhi CodeGuard";
            ClientSize = new Size(720, 480);
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            BackColor = Color.FromArgb(9, 18, 34);
            if (File.Exists(iconPath))
            {
                Icon = new Icon(iconPath);
            }
            if (File.Exists(backgroundPath))
            {
                BackgroundImage = Image.FromFile(backgroundPath);
                BackgroundImageLayout = ImageLayout.Zoom;
            }

            progressBar = new ProgressBar();
            progressBar.Left = 120;
            progressBar.Top = 420;
            progressBar.Width = 480;
            progressBar.Height = 14;
            progressBar.Minimum = 0;
            progressBar.Maximum = 100;
            progressBar.Value = 0;
            Controls.Add(progressBar);
        }

        public void UpdateStatus(string status, int percent, int remainingSeconds)
        {
            if (IsDisposed)
            {
                return;
            }
            int safePercent = Math.Max(0, Math.Min(100, percent));
            progressBar.Value = safePercent;
            Refresh();
        }
    }
}

