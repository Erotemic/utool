# Changelog

We are currently working on porting this changelog to the specifications in
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## Version 2.2.2 - Unreleased


### Version 2.2.1 - Released 2024-08-13

### Changed
* Replace imp usage with importlib.
* Remove most cases of six.moves
* Remove delorean dependency

### [Version 2.2.0] - Released 2024-04-13

### Changed:
* Replaced lena with astro
* Bumped minimum Python to 3.8
* Using new backend for grab-test-image with multiple mirrors and an offline fallback
* Fallback to text mode when pyfiglet cant find a font

### Fixed:
* Removed codecov from test requirements
* Fix utool.Timerit API regression
* Fixed deprecated pipes

### [Version 2.1.7]

### Removed:
* broken `func_callsig` function

### Fixed:
* numpy.int / np.bool issue

### Changed 
* Using loose / strict dependencies
* Fixed issues to support 311


## [Version 2.1.6]

### Fixes:
* Hacked in a fix to `ut.Pref` to prevent certain params from being dynamically
  defined. Ultimately the structure of this class is poorly designed, so I dont
  feel bad about hacking it like this.

* Fixed issue in color text where some colors that were used are no longer
  supported.  Hacked around by changing them to a supported color when an
  exception is thrown.


### Earlier Versions

* Undocumented
