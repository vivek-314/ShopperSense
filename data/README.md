# Dataset provenance

`raw/online_shoppers_intention.csv` is an exact byte copy of the CSV supplied in the original project root. The root file is preserved. No rows were synthesized, removed, reordered or altered in the raw copy. The original download date is unknown.

SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

The observed shape is 12,330 sessions, 17 predictors and Revenue. There are 1,908 purchases and 10,422 non-purchases; 125 excess complete duplicates, preserved. Those dimensions and class counts match the UCI source description. The checksum identifies the supplied file; a new UCI download was not used to assert byte-for-byte canonical identity.

Source: [UCI Online Shoppers Purchasing Intention Dataset, 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset). Sakar, C. & Kastro, Y. (2018). [DOI 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); source metadata checked 16 November 2026.

If the raw file is absent, download the UCI archive and extract `online_shoppers_intention.csv` to `data/raw/`. Compute SHA-256 before running. A different hash may reflect different bytes or line endings: investigate and record the difference; never silently edit a file to make the hash match. The loader records the actual hash, shape and schema for every run.

Stable, zero-based `source_row_id` values are assigned from the original CSV row order after the header. Raw source schema, explicit target mapping, environment and attribution are in `results/data_manifest.json`. Source row IDs and Revenue never enter the model inputs.
