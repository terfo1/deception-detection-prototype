# Assignment 3: Software Development and Integration

## 1. Project overview and research context

Thesis topic: Neural Network Analysis of Real-Time Eye-Tracking Data for Detecting Deceptive Statements. This assignment implements and verifies a small offline scientific analysis module within the existing dissertation repository. The contribution is a reusable path from explicitly identified gaze samples to descriptive features, estimated events, plots and reproducible exports. It does not train a new deception classifier or establish the truth of a statement. The practical result supports inspection of data and methods before a later, properly controlled neural-network experiment.

The repository already contains Python data adapters, preprocessing, trial aggregates, sequence windows, participant-group splitting, baseline classifiers, optional LSTM and TCN models, saved-model inference, tests and GitHub Actions. The baseline dependencies are NumPy, pandas, scikit-learn, Matplotlib, PyYAML and openpyxl; PyTorch and XGBoost are optional. A deterministic agent layer coordinates existing services. The training entry point is scripts/train.py; main.py remains a placeholder. Source data are read from CSV or XLSX, while experiment outputs and checkpoints are stored under ignored output directories. No database or web framework is required for the assignment.

The intended dissertation target is truth/lie, with statements as the proposed decision unit; a suitable raw dataset and final protocol remain unresolved. Concealed Information Test data concern concealed recognition and must remain a separate experimental target. The assignment's new analysis does not use either type of label. The existing real-data adapters and legacy label fallbacks are not validated or repaired by this work. Maintaining this distinction prevents a successful synthetic calculation from being misrepresented as evidence for deception detection.

The repository inspection identified a useful gap: existing aggregates include heuristic fixation counts and permissive temporal fallbacks, rather than a standalone strict event-analysis contract. The new package therefore adds focused functionality without replacing the established training pipeline. Its implementation is located in src/eye_tracking_analysis, with the demonstration in examples/eye_tracking_analysis_demo.py and numerical tests in tests/test_eye_tracking_analysis.py. The report distinguishes verified local behavior from pending remote execution. The updated workflow has not been pushed or run on GitHub, and no participant recordings or heavy research experiments were used.

<!-- page -->

## 2. Software development process

Requirements analysis translated the assignment into observable behaviors. CSV loading must preserve explicit participant, session and trial identifiers; missing metadata must fail rather than be inferred from filenames. Timestamps, when supplied, must be finite and strictly increasing within each group. Coordinate and time units must be declared. Missing gaze must remain distinguishable from observed zero coordinates and explicit blink observations. Tests need predetermined numerical answers, protected exports and operation without a camera, GPU or eye tracker. These requirements define an engineering deliverable while leaving substantive scientific choices open.

System design retained the repository's source-package, configuration, pytest and output conventions. The new package separates validation, analysis, input/output, visualization and CLI concerns. AnalysisConfig makes thresholds, gap policies, optional interpolation, smoothing and screen geometry explicit. AnalysisResult carries processed samples, one feature row per trial, estimated events and metadata. Functions work on pandas tables and return ordinary data structures, allowing reuse without a separate service. The analysis package deliberately has its own strict contract because calling the permissive legacy preprocessing would reintroduce silent defaults and weaker boundaries.

Implementation proceeded from validation to trial-local processing and independently checkable numerical features. The CLI exposes the same functions used by tests and the synthetic example. An installed entry point is also declared in pyproject.toml. The practical calculations use existing libraries and the standard library; no runtime dependency was added. The example is deterministic analytic geometry rather than a stochastic classifier experiment. It produces a fresh run directory and never overwrites an earlier demonstration or scientific result.

Testing combines numerical oracles, invalid-input cases and integration through actual subprocess entry points. Integration updates the existing CI rather than creating another unrelated project or workflow. The core job runs both the existing synthetic training example and the new descriptive analysis, while optional sequence-model checks remain available. Maintenance is supported by explicit limitations, prepared Issues, a README guide, reproducible metadata and the repository's decision/status documents. A future change to event definitions or units should update the contract and associated oracles together. Scientific method validation, physical acquisition and streaming causality remain later tasks; successful software integration does not remove those requirements.

<!-- page -->

## 3. Technology selection

Python was selected because it is already the project's implementation language and supports both scientific analysis and the existing ML pipeline. Its standard library supplies CSV inspection, hashing, JSON, argument parsing and portable paths. R would be a credible alternative for statistics, MATLAB for instrument-oriented signal analysis, and C++ for tightly controlled latency. Migrating the assignment would introduce a second environment without addressing a demonstrated bottleneck. Python's official overview documents its scientific applications and community library ecosystem [1]. The choice is practical continuity, not a claim of universal superiority.

NumPy provides finite-value masks, differences, Euclidean norms, interpolation arithmetic and population variance. These operations express the numerical definitions directly. Python lists would require more manual loops; SciPy is useful for advanced signal filters, but the chosen time-window mean does not require a new direct SciPy dependency. NumPy's documentation is the implementation reference [2]. pandas supplies typed CSV ingestion, identifier preservation, trial grouping and tabular exports. The standard csv module alone would require manual grouping; Polars could support larger tables, but a migration is unnecessary for this small module. Missing values remain explicit rather than being indiscriminately converted to zero [3].

Matplotlib renders standalone PNG figures through the Agg canvas without a GUI [4]. Plotly would support interactive exploration and seaborn richer statistical defaults, but a portable static export satisfies the present requirement with an existing dependency. pytest was retained for numerical assertions, parameterized error cases and temporary-directory fixtures [5]; unittest is built in, while property-based testing could supplement future boundary coverage. pytest-cov generates machine-readable coverage, and Ruff performs the repository's configured coding checks [6]. Neither coverage nor linting measures the validity of the scientific hypothesis.

Git and GitHub Actions continue the repository's established development workflow; their roles are discussed separately below. Existing scikit-learn models, PyYAML configuration, openpyxl input support and optional PyTorch remain available but are not called by the new analysis. Reusing the stack reduces dependency drift and keeps analysis compatible with later ML work. Runtime versions and source hashes are exported so a result can be associated with its actual environment. The declared lower bounds are compatibility constraints, not a lockfile; exact versions should be retained for each substantive experiment.

<!-- page -->

## 4. Version control and collaborative development

Git is a version control system; GitHub is a hosting and collaboration platform. Git records local snapshots, branches and history, while GitHub adds repository access, pull requests, Issues and hosted automation. Distributed version control permits local history and commits without a network connection [7]. The assignment uses the existing repository origin, https://github.com/terfo1/deception-detection-prototype, rather than creating a second repository. The dedicated local branch is feature/assignment-3. No push, public repository creation, issue publication or remote-settings change is part of the completed local work.

The initial working tree contained a user modification to docs/WORK_STATUS.md. That text was preserved. Local commits stage explicit assignment paths and exclude the mixed user-owned status file; generated reports, raw data, checkpoints and environments remain outside commits. Implementation commit 6f9e4be and CI commit 9fada69 separate the analysis/tests from automation. The documentation has a separate commit. Commit messages describe the behavior introduced, while a pull request should state the contract, local validation and pending remote checks. Prepared issue texts in docs/assignment_3/ISSUES.md provide acceptance criteria for annotated-event validation, causal preprocessing and strict legacy training labels.

Apache Subversion is a centralized alternative based around a shared authoritative repository [8]. It can suit organizations with controlled central workflows, but network-independent branching/history is less aligned with this project's local experimentation. Mercurial is another distributed system with local commits and branching [9]; it could satisfy the basic scientific versioning requirement. Git remains appropriate because the project already uses Git and a GitHub origin. Changing systems would add migration and training work without improving this assignment's analysis contract. The comparison concerns workflow fit, not an asserted performance benchmark.

The existing MIT License applies to the software and was not replaced. External datasets retain their own licenses and participant-access restrictions. The ignore rules already protect raw/processed data, outputs, environments and common secrets; temporary Word lock files and temporary files were added. Source control improves traceability only when the source revision, dirty state, configuration and input provenance are also recorded. A branch or green badge alone does not establish that a reported scientific result was generated by the reviewed code. After authorized publication, immutable revision links should replace the pending references in this report.

<!-- page -->

## 5. Continuous integration and delivery

Continuous integration checks proposed changes against a shared set of automated expectations. Continuous delivery adds preparation of releasable artifacts; deployment or publication is a further action. In this project, the configured workflow checks Python code, executes tests and demonstrations, and builds distributions. It does not deploy a live research application. GitHub documents workflow files, event filters, jobs and permissions as part of its Actions syntax [10]. The relevant local configuration is .github/workflows/ci.yml; reading or parsing it is different from observing a successful hosted run.

Pushes to main or feature/assignment-3, pull requests targeting main, version tags and manual dispatch are configured triggers. The core matrix covers Python 3.10 with pandas 2 and Python 3.12 with pandas 3. Each core job checks out the repository, sets up Python, installs core/development dependencies, runs Ruff and non-deep pytest, exports coverage/JUnit reports, and executes both synthetic examples. Figures use a headless backend. These checks do not require a GPU, webcam, physical eye tracker, private dataset or downloaded pretrained checkpoint. The examples generate small data locally.

The optional deep job is restricted to manual dispatch or version tags. It uses CPU PyTorch and tiny regression tests for the existing sequence models; normal assignment pushes and pull requests avoid that dependency download. The package job builds wheel and source distributions. Report and demo artifacts are configured for upload. A GitHub release requires a version tag, successful prerequisites and the separately enabled repository variable ENABLE_RELEASE=true. This explicit opt-in prevents routine assignment verification from becoming publication. No remote variable was changed and no release was created during the task.

For scientific software, CI is valuable because a small parsing or grouping regression can change downstream results without an obvious exception. Here, numerical event tests and boundary cases execute real analysis code. They also protect operation across supported pandas generations. Local YAML parsing and structural assertions confirm the configured triggers and commands, but they cannot verify remote permissions, runner availability, caching or hosted artifacts. The remote run URL remains PENDING_AFTER_AUTHORIZED_PUSH. Continuous delivery here means preparing checked distributions; scientific validation still requires suitable data, a defined protocol and controlled evaluation.

<!-- page -->

## 6. Scientific software development

Reproducibility requires a recorded path from input to output, including methods and environment. Wilson and colleagues recommend organized data, explicit dependencies, reusable functions and tracked changes as practical foundations for research computing [11]. In this assignment, analysis.json records configuration, coordinate/velocity units, input SHA256 for loaded CSV, package versions, module hashes and stated limitations. The synthetic generator records its identity and marks the source synthetic. It has no random draw, so its seed is explicitly null. A new output directory protects earlier runs from accidental replacement.

Experiment tracking should also explain what was evaluated. The demonstration uses one artificial participant, one session and one trial; there is no train/validation/test split, learned transformation, target class or prediction metric. Those fields are not silently omitted from the interpretation: they are inapplicable to a descriptive computation. The accompanying experiment record follows docs/templates/EXPERIMENT_RECORD.md. For future neural-network studies, participant splits must precede fitted preprocessing; all sessions of a participant must stay together. scikit-learn's guidance warns against using test information when fitting transforms [12]. The new optional coordinate normalization uses fixed screen dimensions and learns no statistics.

Data validation is part of the measurement model. Unknown time units, duplicate or decreasing timestamps and missing identifiers cause explicit errors. Long or edge gaze gaps remain unavailable instead of becoming fabricated measurements. Quality measures show original missingness, interpolation and usable samples separately. The estimated sampling frequency is derived from median timestamp intervals and is descriptive; it does not prove that the device used a uniform rate. IDs and trial boundaries are retained, and optional statement identifiers are preserved. Callers must assign separate trials to independent statements.

Research ethics includes protecting participant recordings and avoiding exaggerated inference. Missing gaze is not proof of blinking; estimated eye events are not proof of deception. No new participant data were collected or published. The module does not consume true labels and does not infer classes from filenames. Its offline interpolation uses a future endpoint, so real-time readiness is not claimed. Subsequent experiments need a verified truth/lie protocol, permitted data, explicit decision units and uncertainty at participant level. The assignment improves inspectability and auditability while leaving these scientific responsibilities visible.

<!-- page -->

## 7. Practical implementation: input and preprocessing

The public package interface exposes AnalysisConfig, AnalysisResult, load_csv, analyze and export_results. validation.py enforces the schema; io.py handles CSV and artifacts; analysis.py performs preprocessing and features; visualization.py renders figures; cli.py exposes user parameters. The CSV requires participant_id, session_id, trial_id, gaze_x and gaze_y. Optional timestamp values use explicitly declared seconds or milliseconds. Optional blink values must be complete binary observations. A missing timestamp column permits spatial analysis only; missing values within an existing timestamp column are rejected. Labels and additional columns are preserved without influencing the calculations.

CSV loading checks duplicate headers before pandas can rename them. Identifiers are read as strings to preserve leading zeroes, and a hash identifies the exact input bytes. Validation does not reorder time to hide a faulty acquisition sequence. Each participant/session/trial is processed independently. Milliseconds are converted to seconds once. There is no fabricated time axis based on row order and no inferred label or participant ID. Empty datasets fail with a clear error; wholly missing gaze remains unavailable for spatial features.

A gaze pair is unusable when either coordinate is missing. An explicit blink observation also masks gaze for event analysis. Optional interpolation fills only a whole internal missing run bounded by two valid observations, within the configured elapsed-time limit and without crossing an acquisition gap. The weights use actual timestamps. It never extends a first or last observation into an unobserved edge. Optional smoothing takes a trailing mean over samples in a declared millisecond interval; its point count therefore reflects recorded timing. It resets on unresolved missingness, observed blinks and large timestamp gaps. The mean is a simple noise-reduction method, not a calibrated physiological frequency filter.

Optional pixel normalization divides x by known screen width and y by known screen height. Output units and the velocity threshold are then normalized units and normalized units per second. Pixel-to-degree conversion is not implemented because viewing distance and screen geometry would be required. The default example instead declares degrees directly for artificial coordinates. These choices preserve physical interpretation and prevent a data-fitted normalization from silently changing event thresholds. Interpolation is explicitly offline even though the smoothing window itself looks only backward.

<!-- page -->

## 8. Practical implementation: features and exports

For two adjacent valid samples in the same group, speed is Euclidean coordinate displacement divided by the positive elapsed time. Intervals exceeding the configured acquisition-gap limit are excluded. Consecutive intervals at or below the velocity threshold form a fixation estimate, subject to a minimum duration. Consecutive intervals above it form a saccade estimate. Event duration is the difference between endpoint timestamps. Saccade amplitude is endpoint distance, peak velocity is the largest interval speed, and mean velocity is path length divided by duration. No additional final sample duration is extrapolated.

The approach is intentionally an engineering estimate. Literature on velocity-based identification shows that event thresholds can vary across individuals, tasks and measurements [13]. The implementation does not reproduce the cited adaptive binocular algorithm or claim validated physiological discrimination. Smooth pursuit and noise can challenge a simple threshold. Event validation against permitted reference annotations is a prepared follow-up Issue. Filter and interpolation settings also affect estimates and must remain part of the reported configuration.

Gaze dispersion is the square root of the sum of population variances of processed usable x and y coordinates. No usable observations produce an unavailable value rather than zero dispersion. Blink rate is available only with timestamps and explicit binary blink observations. It counts observed adjacent 0-to-1 transitions per minute of recorded intervals, excluding acquisition gaps. A blink present at the first sample or just after a gap has an unobserved onset and is not counted. Missing gaze alone never supplies a blink flag. Quality fields include input missing fraction, interpolated count, usable samples and acquisition gaps.

Export creates a new directory containing samples.csv, features.csv, events.csv, analysis.json and gaze_analysis.png. Existing directories are rejected. Event CSV retains a stable header even when no events are detected. Figures separate trial traces and break paths at missing data or acquisition gaps. The CLI and example call the same analysis functions as the tests. The analytic demo has 42 samples at 100 Hz, two plateaus, and a 5-degree displacement over 20 ms. Expected outputs are two fixation estimates with mean duration 0.195 s, one saccade with amplitude 5 degrees and peak speed 250 degrees/s, and dispersion 2.469341 degrees. These are predetermined computation checks, not biological findings.

<!-- page -->

## 9. Verification, results and conclusion

The fresh full local pytest run completed with 97 passed in 56.89 seconds on Windows using Python 3.12.4. This includes 35 new analysis tests and all existing tests, including small optional baseline/LSTM/TCN regressions. Numerical coverage includes known 3-4-5 geometry, fixation duration, time-unit equivalence, fixed normalization, time-weighted interpolation and smoothing behavior at different recorded frequencies. Boundary checks cover participants, sessions, trials, missing runs, blinks and acquisition gaps. Invalid-format, empty-data, timestamp and overwrite cases verify refusal behavior. Subprocess integration executes both the documented example and module CLI.

Ruff completed with All checks passed, and pip check reported No broken requirements found. Overall measured source line coverage was 88.61 percent, with 1,681 of 1,897 executable lines covered; this is a software coverage measure, not scientific accuracy. The example executed separately and exported its expected values. A local build produced wheel and source distributions using the existing environment with build --no-isolation. The workflow YAML was parsed and its assignment triggers, analysis-demo command, deep-job condition and release opt-in were checked locally. No successful remote Actions run is asserted.

The first sandbox pytest attempt encountered Windows access errors for temporary directories; a fresh output parent and a permitted run outside the sandbox resolved execution. The first sandbox build likewise failed at its temporary directory and then succeeded outside it. Old temporary paths were not deleted. These are recorded environment limitations, not repaired ML defects. The tested environment included NumPy 2.4.4, pandas 3.0.2, Matplotlib 3.10.8, scikit-learn 1.8.0, pytest 9.1.1 and Ruff 0.16.9. No new analysis runtime dependency was installed.

The assignment delivers a locally demonstrable analysis module, meaningful tests, CI integration, documentation and prepared collaboration artifacts. It supports the dissertation by making input assumptions, estimated eye events and quality inspectable before modeling. Remaining work includes real event validation, strict legacy training labels, appropriate truth/lie data, participant-group evaluation, fully causal preprocessing and physical acquisition. No scientific classifier metric, live latency or deception conclusion is reported. Repository revision link: PENDING_ASSIGNMENT_3_REVISION. Updated workflow run link: PENDING_AFTER_AUTHORIZED_PUSH. Prepared Issue publication and a pull request require a separately authorized external action.

<!-- page -->

## 10. References and implementation evidence

All web references were checked on 8 October 2026. Software documentation describes tools; local source and current checks substantiate this implementation.

[1] Python Software Foundation. About Python. [Official overview](https://www.python.org/about/).

[2] NumPy developers. NumPy documentation. [Official manual](https://numpy.org/doc/stable/).

[3] pandas developers. Working with missing data. [User guide](https://pandas.pydata.org/docs/user_guide/missing_data.html).

[4] Matplotlib developers. Backends. [User guide](https://matplotlib.org/stable/users/explain/figure/backends.html).

[5] pytest developers. Temporary directories and files in tests. [Documentation](https://docs.pytest.org/en/stable/how-to/tmp_path.html).

[6] Astral. Ruff. [Official documentation](https://docs.astral.sh/ruff/).

[7] Chacon, S., and Straub, B. Pro Git, 2nd ed., About Version Control. [Online book](https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control).

[8] Apache Software Foundation. Apache Subversion. [Project overview](https://subversion.apache.org/).

[9] Mercurial project. About Mercurial. [Project overview](https://www.mercurial-scm.org/about).

[10] GitHub. Workflow syntax for GitHub Actions. [Documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).

[11] Wilson, G., et al. (2017). Good enough practices in scientific computing. PLoS Computational Biology, 13(6), e1005510. [doi:10.1371/journal.pcbi.1005510](https://doi.org/10.1371/journal.pcbi.1005510).

[12] scikit-learn developers. Common pitfalls and recommended practices. [User guide](https://scikit-learn.org/stable/common_pitfalls.html).

[13] van der Lans, R., Wedel, M., and Pieters, R. (2011). Defining eye-fixation sequences across individuals and tasks: the Binocular-Individual Threshold (BIT) algorithm. Behavior Research Methods, 43, 239-257. [doi:10.3758/s13428-010-0031-2](https://doi.org/10.3758/s13428-010-0031-2).

Local evidence: [analysis](../../src/eye_tracking_analysis/analysis.py), [CSV/export](../../src/eye_tracking_analysis/io.py), [tests](../../tests/test_eye_tracking_analysis.py), [example](../../examples/eye_tracking_analysis_demo.py), [workflow](../../.github/workflows/ci.yml), [project metadata](../../pyproject.toml), [Issue drafts](ISSUES.md). New files are local until an authorized push. Replace pending revision and run links only with confirmed URLs.
