from attack_detection.phase1_acquire import _choose_prior, _npm_name


def test_clean_pair_uses_nearest_prior_unaffected_version():
    versions = ["1.0.0", "1.1.0", "1.2.0", "1.2.1", "2.0.0"]
    assert _choose_prior(versions, {"1.2.0", "1.2.1"}, {"1.2.1"}) == "1.1.0"


def test_clean_pair_does_not_use_later_release_without_prior_version():
    assert _choose_prior(["2.0.0"], {"1.0.0"}, {"1.0.0"}) == ""


def test_scoped_npm_path_name_is_decoded():
    assert _npm_name("@scope@package") == "@scope/package"
