# Changelog

## [3.0.2](https://github.com/kasperiio/api/compare/v3.0.1...v3.0.2) (2025-10-14)


### Bug Fixes

* release build with poetry ([9018646](https://github.com/kasperiio/api/commit/9018646e739cd9be0a252b3a59636941554a723c))

## [3.0.1](https://github.com/kasperiio/api/compare/v3.0.0...v3.0.1) (2025-10-14)


### Bug Fixes

* release build with poetry ([80c3b31](https://github.com/kasperiio/api/commit/80c3b315a9a88d1db45af8341afa53302f1509fc))

## [3.0.0](https://github.com/kasperiio/api/compare/v2.2.2...v3.0.0) (2025-10-14)


### ⚠ BREAKING CHANGES

* Nordpool provider has been removed as it now requires authentication

### Code Refactoring

* remove Nordpool provider and simplify data fetching ([f014853](https://github.com/kasperiio/api/commit/f014853c4a57c5519c79f656ee63a158d161dff1))

## [2.2.2](https://github.com/kasperiio/api/compare/v2.2.1...v2.2.2) (2025-09-02)


### Bug Fixes

* provider data availabilty filter based on europe/stockholm instead of utc ([c314411](https://github.com/kasperiio/api/commit/c3144110860b56988c307e4a798ab120b47feb5c))

## [2.2.1](https://github.com/kasperiio/api/compare/v2.2.0...v2.2.1) (2025-09-01)


### Bug Fixes

* latest endpoint start time calculation with timezone ([e416cca](https://github.com/kasperiio/api/commit/e416cca02000fd82f38e777f22e193d4ea61650a))

## [2.2.0](https://github.com/kasperiio/api/compare/v2.1.0...v2.2.0) (2025-09-01)


### Features

* 15 minute interval support ([97b5a2e](https://github.com/kasperiio/api/commit/97b5a2ed222f60f10928eb5cf9990087b8f70f5e))

## [2.1.0](https://github.com/kasperiio/api/compare/v2.0.0...v2.1.0) (2025-08-31)


### Features

* latest endpoint, get todays and tomorrows data ([4b5d4f1](https://github.com/kasperiio/api/commit/4b5d4f1e14769bef3fd5c83737bd23a08e9a5f20))

## [2.0.0](https://github.com/kasperiio/api/compare/v0.1.3...v2.0.0) (2025-08-31)


### Miscellaneous Chores

* v2 ([fb23bce](https://github.com/kasperiio/api/commit/fb23bced23323ee175ab143ef3fc94d8a9727792))

## [0.1.3](https://github.com/kasperiio/api/compare/v0.1.2...v0.1.3) (2025-08-31)


### Reverts

* docker build escaped dots ([4650e97](https://github.com/kasperiio/api/commit/4650e9748cf9497c6a68996ec0a77cb814e60288))

## [0.1.2](https://github.com/kasperiio/api/compare/v0.1.1...v0.1.2) (2025-08-31)


### Bug Fixes

* naive timezones treated as timezone_str, code cleanup ([38009eb](https://github.com/kasperiio/api/commit/38009ebc049fc9a1b92ace9726781a841c6ae88f))

## [0.1.1](https://github.com/kasperiio/api/compare/v0.1.0...v0.1.1) (2025-08-31)


### Bug Fixes

* db cache incorrect due to nordpool using stockholm tz ([28eda8c](https://github.com/kasperiio/api/commit/28eda8cd1c00eca95ad98ce963c9b3bfe02bd01e))

## [0.1.0](https://github.com/kasperiio/api/compare/v0.0.14...v0.1.0) (2025-08-31)


### ⚠ BREAKING CHANGES

* use utc times throught, api handles conversion

### Features

* nordpool api prioritization ([6d3b7c9](https://github.com/kasperiio/api/commit/6d3b7c9b8a507e988ec10492131814a76b634867))


### Code Refactoring

* use utc times throught, api handles conversion ([af6f80a](https://github.com/kasperiio/api/commit/af6f80a870c9b0f5c7b5611697202ea9736360ac))

## [0.0.14](https://github.com/kasperiio/api/compare/v0.0.13...v0.0.14) (2024-11-11)


### Bug Fixes

* actually fix wrong schema and docker build automation ([e34b14f](https://github.com/kasperiio/api/commit/e34b14f933587aa4842f41fe00a3e7512e9b57c2))

## [0.0.13](https://github.com/kasperiio/api/compare/v0.0.12...v0.0.13) (2024-11-11)


### Bug Fixes

* allow negative price ratio ([0308406](https://github.com/kasperiio/api/commit/030840619e142c13226335623dd7a793551845c6))
* trigger release build with release creation ([ac88887](https://github.com/kasperiio/api/commit/ac88887237c84c9c85939a91936217df5f53ec37))

## [0.0.12](https://github.com/kasperiio/api/compare/v0.0.11...v0.0.12) (2024-11-11)


### Bug Fixes

* remove date range limiter ([5270241](https://github.com/kasperiio/api/commit/52702416ddd04f7c82403506e45c48ce89ba0a41))

## [0.0.11](https://github.com/kasperiio/api/compare/v0.0.10...v0.0.11) (2024-11-11)


### Bug Fixes

* router prefix included twice ([461fc84](https://github.com/kasperiio/api/commit/461fc8406a60cda8135e01c28aceaf8784383712))

## [0.0.10](https://github.com/kasperiio/api/compare/v0.0.9...v0.0.10) (2024-11-11)


### Miscellaneous Chores

* release 0.0.10 ([5269472](https://github.com/kasperiio/api/commit/526947266bfd7f28948b2746691dd07051e91d55))

## [0.0.9](https://github.com/kasperiio/api/compare/v0.0.8...v0.0.9) (2024-10-27)


### Bug Fixes

* handle negative ratio calculation better ([0694d66](https://github.com/kasperiio/api/commit/0694d6673e6f96094e56f49b50efd3de409573a6))

## [0.0.8](https://github.com/kasperiio/api/compare/v0.0.7...v0.0.8) (2024-10-20)


### Bug Fixes

* fully negative day resulted in bad ratio calculation ([573ca90](https://github.com/kasperiio/api/commit/573ca903249187cc0e03973eb39360c189d15cad))

## [0.0.7](https://github.com/kasperiio/api/compare/v0.0.6...v0.0.7) (2024-10-04)


### Bug Fixes

* get namespace dynamically from entso-e response ([a48ff07](https://github.com/kasperiio/api/commit/a48ff079599476e9db41b00236cfc12152cf0bf5))
* take only 00-23 into account when calculating ratio ([08d30cb](https://github.com/kasperiio/api/commit/08d30cbe5c7f57420b3fec2aeeec266e4a511c20))

## [0.0.6](https://github.com/kasperiio/api/compare/v0.0.5...v0.0.6) (2024-09-24)


### Bug Fixes

* use workflow dispatch to trigger docker build ([2d056b3](https://github.com/kasperiio/api/commit/2d056b393b710cce5682968169d32eddbcd8517c))

## [0.0.5](https://github.com/kasperiio/api/compare/v0.0.4...v0.0.5) (2024-09-24)


### Bug Fixes

* docker publish, hopefully? ([eb03af1](https://github.com/kasperiio/api/commit/eb03af1e16315c1354f6fcb39d70ee6aa85800c4))

## [0.0.4](https://github.com/kasperiio/api/compare/v0.0.3...v0.0.4) (2024-09-23)


### Bug Fixes

* faulty logic in ratio property ([c4158c9](https://github.com/kasperiio/api/commit/c4158c90461c6044fe4ec958b8d165fe77e49f54))

## [0.0.3](https://github.com/kasperiio/api/compare/v0.0.2...v0.0.3) (2024-09-23)


### Bug Fixes

* conventional commits workflow ([13ca12e](https://github.com/kasperiio/api/commit/13ca12e37b6b2e9c5d6c845c33546df6694d0758))
* docker publish yaml ([944b6d3](https://github.com/kasperiio/api/commit/944b6d30f2abd465cbd083e87a54c84135494763))

## [0.0.2](https://github.com/kasperiio/api/compare/v0.0.1...v0.0.2) (2024-09-23)


### Miscellaneous Chores

* release 0.0.2 ([2ff95e2](https://github.com/kasperiio/api/commit/2ff95e2d3f62750a93cd2a3d896857eebc60a3a6))

## 0.0.1 (2024-09-23)


### Miscellaneous Chores

* release 0.0.1 ([8ef340c](https://github.com/kasperiio/api/commit/8ef340ceeda11eaccaa5258be957f5c39db6667b))
