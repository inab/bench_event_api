pipeline {
    agent any
    
        stages {

            stage('activate virtual environment') {
                steps {
                    // Install Modules
                    bash 'source .pyenv/bin/activate'
                }
            }

            stage('Build') {
                steps {
                    // Create dist folder
                    bash 'python app.py'
                }
            }     
        }
}
