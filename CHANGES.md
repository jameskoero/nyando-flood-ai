After studying the Copilot-identified issues, I have:
  - Documented WHY each function exists with hydrological reasoning
  - Explained the BOM header problem from GEE exports
  - Added CHANGES.md summarising the 4 root causes I found
  - Annotated SMOTE rewrite: why pandas not numpy
  - Documented pickle-vs-joblib decision
  - Explained canonical CSV pinning in test_pipeline.py

All source changes from the previous two commits are preserved;
this commit adds understanding-level documentation on top.
