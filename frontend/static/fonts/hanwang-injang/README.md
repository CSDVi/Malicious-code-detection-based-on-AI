# HanWang InJang font assets

- `HanWangInJang.ttf` is the original font supplied with the project.
- `HanWangInJang-XiezhiBrand.woff2` is the web subset used by the dashboard brand title.
- The source font is a Traditional Chinese font and does not contain simplified `码` (U+7801). The web subset maps the source `碼` outline to U+7801 while the page keeps the accessible text `獬豸安码`.
- The source maps `獬` and `豸` to empty glyphs. Those two characters therefore fall back to the previously bundled calligraphy font instead of disappearing.
