# Adema
This is the related research scripts and dataset for the paper "Automated Discovery and Mapping ATT&amp;CK Tactics and Techniques for Unstructured Cyber Threat Intelligence"


# Files
## Datasets
### TRAM
- `tram_original.json`: The raw file copy from [TRAM](https://github.com/center-for-threat-informed-defense/tram/tree/main/data).
- `tram_original_with_labels.json`: The original TRAM dataset with technique names converted into technique IDs.
- `tram_with_all_labels.csv`: Text samples built from TRAM dataset with all labels annonated.

### ATT&amp;CK
- `attack_original.csv`: Text samples from ATT&amp;CK web pages with single label annonated.
- `attack_with_all_labels.csv`: Text samples integrated from the single label file with all labels annonated.

## Scripts
- `collector.py`: Tactics, techniques and text samples collector for ATT&amp; web pages.
- `main.py`: Code for experiments.