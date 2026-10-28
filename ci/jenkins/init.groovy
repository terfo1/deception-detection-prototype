import jenkins.model.Jenkins
import hudson.security.HudsonPrivateSecurityRealm
import hudson.security.FullControlOnceLoggedInAuthorizationStrategy
import hudson.plugins.git.GitSCM
import hudson.plugins.git.UserRemoteConfig
import hudson.plugins.git.BranchSpec
import org.jenkinsci.plugins.workflow.job.WorkflowJob
import org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition
import hudson.triggers.SCMTrigger

// Bootstrap only an isolated assignment controller; never reset an existing user.
def instance = Jenkins.get()
if (instance.getSecurityRealm() == hudson.security.SecurityRealm.NO_AUTHENTICATION) {
    def password = System.getenv('JENKINS_ADMIN_PASSWORD')
    if (!password || password.length() < 20) {
        throw new IllegalArgumentException('JENKINS_ADMIN_PASSWORD must have at least 20 characters')
    }
    def realm = new HudsonPrivateSecurityRealm(false)
    realm.createAccount('research-admin', password)
    instance.setSecurityRealm(realm)
    def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
    // Read-only dashboard for screenshots; the port is bound to 127.0.0.1 only.
    strategy.setAllowAnonymousRead(System.getenv('JENKINS_ANONYMOUS_READ') == 'true')
    instance.setAuthorizationStrategy(strategy)
}
instance.setNumExecutors(1)
instance.setLabelString('assignment5-local')
def jobName = 'eye-tracking-assignment5'
if (instance.getItem(jobName) == null) {
    def job = instance.createProject(WorkflowJob, jobName)
    def repository = System.getenv('RESEARCH_REPOSITORY')
    def branch = System.getenv('RESEARCH_BRANCH') ?: 'main'
    def scm = new GitSCM([new UserRemoteConfig(repository, null, null, null)],
        [new BranchSpec('*/' + branch)], false, [], null, null, [])
    def definition = new CpsScmFlowDefinition(scm, 'Jenkinsfile')
    definition.setLightweight(true)
    job.setDefinition(definition)
    job.addTrigger(new SCMTrigger('H/2 * * * *'))
    job.save()
}
instance.save()
