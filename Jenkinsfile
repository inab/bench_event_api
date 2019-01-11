pipeline {
    agent any
    
        stages {

            stage('activate virtual environment') {
                steps {
                    // Install Modules
                    sh 'source .pyenv/bin/activate'
                }
            }

            stage('Build') {
                steps {
                    // Create dist folder
                    sh 'python app.py'
                }
            }     
        }
}
