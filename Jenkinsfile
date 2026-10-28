pipeline {
    agent any
    options {
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '15'))
        timestamps()
    }
    // Works behind NAT without exposing a webhook. Coalesces commits between polls.
    triggers { pollSCM('H/2 * * * *') }
    environment { DOCKER_BUILDKIT = '1' }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script { env.REVISION = sh(script: 'git rev-parse HEAD', returnStdout: true).trim() }
                echo "Revision ${env.REVISION}"
            }
        }
        stage('Build test image') {
            steps {
                sh 'docker build --pull --target test --build-arg VCS_REF=$REVISION -t eye-tracking-ci:$BUILD_NUMBER .'
            }
        }
        stage('Unit tests and quality gates') {
            steps {
                sh '''
                    set -eu
                    mkdir -p test-results
                    docker create --name eye-ci-$BUILD_NUMBER eye-tracking-ci:$BUILD_NUMBER
                    status=0
                    docker start -a eye-ci-$BUILD_NUMBER || status=$?
                    docker cp eye-ci-$BUILD_NUMBER:/app/test-results/. test-results/
                    docker rm eye-ci-$BUILD_NUMBER
                    exit "$status"
                '''
            }
            post { always { junit testResults: 'test-results/junit.xml', allowEmptyResults: false } }
        }
        stage('Build runtime image') {
            steps {
                sh 'docker build --target runtime --build-arg VCS_REF=$REVISION -t eye-tracking:$BUILD_NUMBER .'
            }
        }
        stage('Container smoke test') {
            steps {
                sh '''
                    set -eu
                    docker create --name eye-smoke-$BUILD_NUMBER eye-tracking:$BUILD_NUMBER --output-dir /app/outputs/smoke
                    docker start -a eye-smoke-$BUILD_NUMBER
                    docker cp eye-smoke-$BUILD_NUMBER:/app/outputs/smoke test-results/synthetic-smoke
                    docker rm eye-smoke-$BUILD_NUMBER
                '''
            }
        }
        stage('Record provenance') {
            steps {
                sh '''
                    echo "$REVISION" > test-results/revision.txt
                    docker image inspect eye-tracking:$BUILD_NUMBER > test-results/runtime-image.json
                    docker image inspect eye-tracking-ci:$BUILD_NUMBER > test-results/test-image.json
                    docker version > test-results/docker-version.txt
                '''
            }
        }
        stage('Push to lab registry') {
            when { expression { env.LAB_REGISTRY?.trim() } }
            steps {
                script { env.LAB_IMAGE = "${env.LAB_REGISTRY}/eye-tracking:${env.REVISION}" }
                sh '''
                    set -eu
                    docker tag eye-tracking:$BUILD_NUMBER "$LAB_IMAGE"
                    docker push "$LAB_IMAGE" | tee test-results/lab-registry-push.txt
                '''
            }
        }
        stage('Push to Docker Hub') {
            // Optional: set DOCKERHUB_REPOSITORY (user/repo) and add a "dockerhub" credential.
            when { expression { env.DOCKERHUB_REPOSITORY?.trim() } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub',
                        usernameVariable: 'HUB_USER', passwordVariable: 'HUB_TOKEN')]) {
                    sh '''
                        set -eu
                        echo "$HUB_TOKEN" | docker login --username "$HUB_USER" --password-stdin
                        docker tag eye-tracking:$BUILD_NUMBER "$DOCKERHUB_REPOSITORY:$REVISION"
                        docker tag eye-tracking:$BUILD_NUMBER "$DOCKERHUB_REPOSITORY:latest"
                        docker push "$DOCKERHUB_REPOSITORY:$REVISION"
                        docker push "$DOCKERHUB_REPOSITORY:latest"
                        docker logout
                    '''
                }
            }
        }
        stage('Deploy to Kubernetes test') {
            when {
                expression {
                    env.LAB_IMAGE && env.KUBECONFIG_SOURCE?.trim() &&
                        sh(script: 'test -r "$KUBECONFIG_SOURCE"', returnStatus: true) == 0
                }
            }
            steps {
                sh '''
                    set -eu
                    mkdir -p test-results/k8s
                    sed "s#https://127.0.0.1:6443#$KUBE_API_SERVER#" "$KUBECONFIG_SOURCE" > "$WORKSPACE/.kubeconfig"
                    export KUBECONFIG="$WORKSPACE/.kubeconfig"
                    kubectl apply -f k8s/namespace.yaml
                    kubectl -n eye-tracking-test delete job synthetic-analysis --ignore-not-found --wait=true
                    sed -e "s#__IMAGE__#$LAB_IMAGE#" -e "s#__REVISION__#$REVISION#" k8s/analysis-job.yaml \
                        > test-results/k8s/analysis-job.rendered.yaml
                    kubectl apply -f test-results/k8s/analysis-job.rendered.yaml
                    status=0
                    kubectl -n eye-tracking-test wait --for=condition=complete --timeout=240s job/synthetic-analysis || status=$?
                    kubectl -n eye-tracking-test logs job/synthetic-analysis > test-results/k8s/job.log || true
                    kubectl -n eye-tracking-test get job,pods -o wide > test-results/k8s/resources.txt || true
                    kubectl version > test-results/k8s/kubectl-version.txt || true
                    if [ "$status" -ne 0 ]; then
                        kubectl -n eye-tracking-test describe job/synthetic-analysis || true
                        exit "$status"
                    fi
                    grep -q "SYNTHETIC DATA" test-results/k8s/job.log
                    cat test-results/k8s/resources.txt
                '''
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'test-results/**', allowEmptyArchive: true, fingerprint: true
            sh 'rm -f .kubeconfig; docker rm -f eye-ci-$BUILD_NUMBER eye-smoke-$BUILD_NUMBER 2>/dev/null || true'
        }
    }
}
