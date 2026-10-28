# The Role of DevOps and CI/CD in Automating Eye-Tracking Research

Assignment 5 - theoretical report | 28 October 2026

## 1. DevOps goals, principles and research benefits

DevOps brings software development and operation into a shared feedback process. A team versions the application together with its execution environment, automates repeatable checks and makes failures visible. The aim is reliable, reviewable change. In a research team, the consumers include the experimenter, another student reproducing a result and a future maintainer. All need to know which code and environment produced an artifact. DORA describes frequent integration, small changes and rapid automated feedback as central CI practices [1].

This project investigates neural analysis of eye-tracking recordings associated with deceptive statements. Its implemented tools include offline gaze analysis, classical models, optional LSTM/TCN models and synthetic demonstrations. Physical live acquisition and a completely causal processing pipeline remain future work. Truth/lie at statement level and concealed recognition in a Concealed Information Test are different targets. Infrastructure must preserve that distinction instead of giving both experiments an apparently comparable accuracy score.

A typical reproducibility failure is a change in a table library that silently removes participant identifiers during grouping. A model may still train while the intended experimental units disappear. Another failure is treating unavailable labels as a default class. Automated checks should therefore exercise identifiers, trial boundaries, explicit class handling and grouping, in addition to checking whether an executable starts. Existing permissive legacy label fallbacks remain a documented limitation; adding DevOps does not repair them automatically.

The practical principles are shared responsibility, version control, environment declaration, automated quality gates and traceability. A proposed change should have a small diff and a reviewable test result. Raw recordings, credentials and generated scientific results belong outside the image and source history. A run should record its revision, configuration, dependency versions and source identity. These practices reduce accidental variation and help separate a software defect from a scientific disagreement.

For this assignment, the research code is preserved while its delivery process is extended. The lightweight offline analysis provides a suitable container demonstration because it needs no hardware, licensed dataset or expensive training. Synthetic geometry verifies software execution only; it cannot establish deception-detection accuracy.

<!-- page -->

## 2. Continuous integration and continuous delivery

Continuous integration combines frequent changes with automated builds and checks. A failed check blocks progression to later delivery stages. Continuous delivery maintains a tested artifact that can be deployed when a person approves its use; continuous deployment additionally releases qualifying changes automatically. In a research setting, a tested container is a deliverable, whereas approval to run a human-subject study or publish a scientific conclusion remains a separate decision.

The proposed pipeline is: checkout a Git revision; build a test image; run dependency checks, lint, core tests and package construction; build the runtime image; execute a synthetic container smoke test; archive reports and provenance. A common shell entry point keeps the container quality gates aligned between Jenkins, GitHub Actions and local execution. Test and runtime stages share the same base environment, reducing differences between what was checked and what is delivered.

Fast checks belong on ordinary commits. Optional sequence-model tests are more expensive and require PyTorch, so the existing dedicated manual/tag workflow remains separate. The default container intentionally excludes that dependency. This makes its scope explicit: a successful core build verifies the offline analysis and core software paths, not every neural-network configuration, GPU stack or physical device.

Useful outputs include JUnit test results, coverage XML, the resolved package list, built distributions, image identifiers and the checked-out revision. Jenkins attaches test results to a build and archives artifacts. Coverage identifies unexecuted paths but does not certify their scientific validity. A high aggregate percentage can coexist with a missing causality test or incorrect participant split; test meaning matters more than the number alone.

For the local lab, Jenkins polls Git every two minutes because the controller is bound to loopback and is not reachable by a public webhook. Polling can combine several commits into one latest-revision build [2]. The GitHub Actions push workflow adds automatic hosted checks. Exact one-build-per-commit operation would require an event queue or webhook and an always-available agent; this lab should not claim that guarantee. Each actual build still identifies the precise revision it tested.

Continuous delivery here produces a smoke-tested image tagged with the Git revision, pushes it to a private lab registry and deploys it to a Kubernetes test namespace. Publication to Docker Hub is implemented as a stage that runs only when a repository name and a Jenkins credential are configured. Keeping external publication opt-in leaves the experimenter in control of which model or result becomes public.

<!-- page -->

## 3. Docker, Jenkins and Kubernetes in scientific work

Docker packages the interpreter, dependencies and application into an image. For this project, a Python 3.12 Linux image contains NumPy, pandas, scikit-learn and Matplotlib. Separate test and runtime targets keep developer tools out of the runtime layer. The application runs under a non-root user. An allow-list in .dockerignore prevents local recordings, model outputs, Git credentials and large archives from entering the build context. These choices follow Docker's guidance on stage separation and limiting build contents [3].

An exact base-image digest and explicit package versions reduce environment drift. The image is still not a complete scientific record: input permissions, experimental labels, seed, split manifest and model specification must also be preserved. Rebuilding from a network package index is not equivalent to an offline, hash-verified supply chain. A stronger future release could archive approved wheels and verify their hashes, then deliberately update them when dependencies need maintenance.

Jenkins coordinates the steps through a versioned Jenkinsfile. A researcher can inspect which stage failed instead of reconstructing an interactive shell session. In this assignment, a local controller uses a dedicated Docker-in-Docker daemon over TLS, with named volumes for its state. The UI is exposed only on the local machine. Jenkins documents this container deployment approach [4]. A private lab can accept this privileged daemon; an institutional service should use isolated agents and a reviewed authentication policy.

A concrete Docker example is executing the known 3-4-5 gaze displacement demonstration on a colleague's Linux computer. A Jenkins example is rejecting a commit that breaks trial-local processing before a new image is built for delivery. Both examples address engineering reproducibility. Neither demonstrates that gaze movement is a valid indicator of a deliberately false statement.

Kubernetes schedules containers across machines. Its Job controller is appropriate for finite tasks, such as running a frozen model on several approved recordings or evaluating predeclared participant folds [5]. Each fold must receive its own output location and keep all sessions of a participant in one group. Cluster scheduling does not create statistically independent participants or prevent leakage by itself. Resource requests and access controls would also be needed before using sensitive data.

In this project a single-node k3s cluster acts as the test environment. Jenkins applies a Job manifest that runs the revision-tagged image as a non-root container with a read-only root filesystem, no service-account token and resource limits, then waits for completion and archives the logs. The cluster verifies deployability, not a new scientific claim. Docker handles the environment, Jenkins the workflow, Kubernetes the execution and Git identifies the code; their roles are complementary.

<!-- page -->

## 4. Conclusion: what DevOps solves in scientific teams

Automation addresses inconsistent environments, forgotten validation steps, opaque artifacts and slow discovery of regressions. It also improves handover: the next researcher can read a pipeline and rerun its commands. The strongest practical result is a chain from source revision to checks, image identity and exported evidence. The assignment implements that chain for the existing offline eye-tracking prototype.

Infrastructure reliability and scientific validity require different evidence. A passing synthetic example does not establish accuracy on new participants. A container cannot make whole-record normalization causal, and replay cannot establish the physical origin of samples. Before claiming a streaming research instrument, the project still needs verified truth/lie recordings, strict training labels, causal preprocessing, future-influence and chunk-size tests, participant-grouped evaluation and separate accumulation/compute latency measurements.

Pipeline health should be measured by build outcomes, time to feedback and reproducible artifact retrieval. Scientific evaluation should separately report the unit of labeling, class meaning, participant grouping, signal quality, abstentions and uncertainty. Combining these into a single green status hides the remaining research questions. DevOps is valuable precisely when it makes those boundaries explicit and preserves negative results as carefully as successful ones.

The recommended next step is to keep the core checks fast, maintain a small approved container environment and add scientifically meaningful tests as the causal and data-contract work proceeds. Large training jobs should remain explicit experiments with provenance rather than execute on every documentation commit. In this form, DevOps supports an honest, maintainable research workflow while leaving scientific acceptance to suitable data and experiments.

## References

[1] DORA. Continuous integration. https://dora.dev/capabilities/continuous-integration/

[2] Jenkins. Pipeline syntax: triggers and post conditions. https://www.jenkins.io/doc/book/pipeline/syntax/

[3] Docker. Building best practices. https://docs.docker.com/build/building/best-practices/

[4] Jenkins. Installing Jenkins with Docker. https://www.jenkins.io/doc/book/installing/docker/

[5] Kubernetes. Jobs. https://kubernetes.io/docs/concepts/workloads/controllers/job/

Official documentation consulted on 28 October 2026. Project-specific statements were checked against the local source and documented limitations.
