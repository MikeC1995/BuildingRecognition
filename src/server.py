import saveable_matcher
matcher = saveable_matcher.SaveableFlannBasedMatcher("wills")
matcher.test()
matcher.load()
