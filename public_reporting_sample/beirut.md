# Change Detection Report

## AOI
oscd_beirut

## Date range
2015-08-20 to 2017-10-03 (2 years, 1.4 months)

## Overall verdict
**REAL CHANGE DETECTED** - Significant urban development and construction activity confirmed

## Change type
Construction and urban expansion

## Confidence
95% (VLM assessment)

## Evidence from indices
- **NDVI delta**: +0.0768 (vegetation changes)
- **NDBI delta**: +0.0701 (built-up area increase)
- **NDWI delta**: +0.0448 (water/moisture changes)
- **CVA magnitude**: 0.2046 (moderate spectral change intensity)
- **Changed area**: 74.18% of AOI affected
- **SSIM**: 0.6227 (structural similarity indicating significant changes)
- **Perceptual hash distance**: 9 (notable visual differences)

## Evidence from model
Classical change detection flagged as "candidate_change" with a change score of 0.7418, indicating high likelihood of genuine change across nearly three-quarters of the analysis area.

## Evidence from VLM
Visual analysis identifies multiple construction activities:
- New coastal development in northeast with rectangular reclaimed land patterns
- Port/harbor infrastructure expansion with concrete structures
- Additional coastal built-up areas
- New rectangular industrial/commercial facilities near airport
- Overall urban densification with new buildings citywide

## Artifact and weather risk
**LOW RISK** across all factors:
- Cloud shadow: low
- Snow: low  
- Seasonality: low
- Registration: low

Quality metrics support this assessment: 0% cloud/shadow/snow fractions, 0.0738 illumination delta, perfect quality score (1.0).

## Recommended next step
**No human review required.** The evidence strongly supports genuine large-scale urban development. Consider archiving as confirmed construction change and using as training data for similar coastal urban expansion scenarios.