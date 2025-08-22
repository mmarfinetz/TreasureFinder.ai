

<taskmaster>
  <title>Fix Earth Engine computePixels and sampling failures in satellite project</title>

  <agents>
    <agent id="satellite-geology-analyzer" role="Geospatial Data & Earth Engine Specialist" />
    <agent id="ml-satellite-engineer" role="ML/CV Engineer for satellite pipelines" />
  </agents>

  <context>
    <repo_root>/Users/mitch/Desktop/Organized/Compare_Satellite_scripts</repo_root>

    <observed_errors>
      - computePixels failed: "Cannot load file containing pickled data when allow_pickle=False"
      - Sampling failed: "Collection query aborted after accumulating over 5000 elements."
      - Fallback used: "Using reduced Earth Engine data with spatial modeling"
      - Coordinates triggering issue (sample):
        (-13.1631, -72.5450), (-13.1586, -72.5434), (-13.1556, -72.5390),
        (-13.1553, -72.5328), (-13.1584, -72.5261), (-13.1651, -72.5207),
        (-13.1745, -72.5182), (-13.1856, -72.5199)
    </observed_errors>

    <key_files>
      - treasure_hunter_module.py
      - satellite_production_modular_unified.ipynb (converted via convert_notebook.py)
      - CLAUDE.md
      - ml_engineer_prompt.txt
      - DEPLOYMENT.md
    </key_files>

    <reference_policies>
      - Preserve existing structures and unrelated code.
      - Use explicit variable names and robust error handling.
      - Add tests for functional changes.
      - Prefer modular edits and performance/security considerations.
    </reference_policies>
  </context>

  <goals>
    - Diagnose and resolve Earth Engine data acquisition failures (computePixels and sampling).
    - Ensure resilient, performant fetching with cloud masking, collection size constraints, and robust fallbacks.
    - Eliminate NumPy pickle-related load errors.
    - Provide automated checks to verify the fix.
  </goals>

  <constraints>
    - No apologies; no unnecessary summaries.
    - File-by-file edits; keep changes minimal and targeted.
    - Add unit/integration checks where functionality changes.
    - Maintain compatibility with existing dependencies and deployment.
  </constraints>

  <tools>
    - Use the Task Tool to create, assign, and track tasks.
    - Use repo-aware code edits exactly where needed.
  </tools>

  <plan>
    1) Reproduce error locally via a minimal call path and capture exact stack traces.
    2) Audit Earth Engine request path for computePixels/sampling:
       - Limit ImageCollection size with filterDate/filterBounds/cloud filters.
       - Reduce to a single image via quality sort + first(), median(), or mosaic().
       - Ensure appropriate region/scale and maxPixels usage where applicable.
       - Implement cloud masking (QA60 / s2cloudless) and composite.
    3) Investigate NumPy pickle error:
       - Search for numpy.load usage and any pickle-dependent arrays.
       - Replace with safe loaders (np.frombuffer/np.asarray) for raw bytes, or proper decoding for EE responses.
       - Ensure no serialized pickled binaries are loaded.
    4) Add safeguards:
       - Hard caps on collection size (e.g., .limit()) and spatial bounds.
       - Retries with backoff; fallbacks to alternative data paths.
    5) Add automated checks:
       - Functional test against the failing coordinates to ensure success or well-structured fallback result.
       - Log improvements and metrics (collection size, cloud %, method used).
    6) Document new behavior in CLAUDE.md.
  </plan>

  <tasks>
    <task id="T1" title="Reproduce and capture stack traces for failing coordinates" assignee="satellite-geology-analyzer" />
    <task id="T2" title="Audit computePixels/sampling pipeline and add caps & masks" assignee="satellite-geology-analyzer" />
    <task id="T3" title="Fix NumPy pickle error source and replace unsafe loads" assignee="ml-satellite-engineer" />
    <task id="T4" title="Implement robust fallbacks and retries in fetch path" assignee="ml-satellite-engineer" />
    <task id="T5" title="Add functional check for failing coordinates" assignee="ml-satellite-engineer" />
    <task id="T6" title="Update CLAUDE.md with new guidance and commands" assignee="satellite-geology-analyzer" />
  </tasks>

  <execution>
    <instructions>
      - Use the Task Tool:
        1. Create tasks T1–T6 with titles above.
        2. Start with T1 and T2 in parallel; others pending.
        3. After completing each task, update status and attach diffs/commands used.
      - Investigate code in:
        - treasure_hunter_module.py (fetch_satellite_image, cloud mask, sampling paths, any numpy.load usage)
        - Converted module from satellite_production_modular_unified.ipynb where EE logic lives.
      - For sampling abort problem:
        - Apply: filterDate, filterBounds, CLOUDY_PIXEL_PERCENTAGE < 20 (QA60 or s2cloudless), .limit(N), and reduce ImageCollection via .median(), .mosaic(), or sorted .first().
        - Ensure region window and scale are reasonable; if necessary, tile the region and stitch results.
      - For pickle error:
        - Search for numpy.load(…) calls; avoid allow_pickle=True.
        - If loading EE responses, decode via standard array types or frombuffer; do not load unknown .npy/.npz from network or untrusted streams.
      - Add a concise functional test that:
        - Calls the same code path for one of the failing coordinates.
        - Asserts result object structure, non-exception, and records method used (computePixels/sampling/reduced fallback).
      - Keep edits modular and minimal; add logging for collection sizes and selected method.
    </instructions>
  </execution>

  <reporting>
    <status_format>
      Provide compact updates:
      - current_task: T#
      - action: what changed
      - files: [paths]
      - result: pass/fail + metrics (collection size, cloud %, method)
      - next: T# to start
    </status_format>
  </reporting>

  <acceptance_criteria>
    - No computePixels pickle-related errors.
    - No sampling abort errors for the provided coordinates (either success or clean fallback).
    - ImageCollection bounded (filters + limit) and cloud masking applied.
    - Logs include: provider method chosen, collection size, cloud %, and timing.
    - Functional test passes locally.
    - Documentation updated in CLAUDE.md with new guidance and commands.
  </acceptance_criteria>

  <handoff>
    - Output: list of edits with file paths, brief rationale per edit, test results, and updated documentation section.
  </handoff>
</taskmaster>
