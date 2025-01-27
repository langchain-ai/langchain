"""Tool for the Jenkins API"""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.jenkins import JenkinsAPIWrapper


class JenkinsSchema(BaseModel):
    """Input for the Jenkins tool.
    
    
        Instantiate:

            .. code-block:: python
        
                    from tools.jenkins.tool import JenkinsJobRun
                    from tools.jenkins.utility import JenkinsAPIWrapper

                    tools = [JenkinsJobRun(
                        api_wrapper=JenkinsAPIWrapper(
                            jenkins_server="https://jenkinsserver.com",
                            username="admin",
                            password=os.environ["PASSWORD"]
                        )
                    )]

        Invoke directly with args:

            .. code-block:: python

                # delete jenkins job
                tools[0].invoke({'job': "job01", "action": "delete"})

                # create jenkins job
                jenkins_job_content = ""
                src_file = "job1.xml"
                with open(src_file) as fread:
                    jenkins_job_content = fread.read()
                tools[0].invoke({'job': "job01", "config_xml": jenkins_job_content,
                                "action": "create"})

                # run the jenkins job
                tools[0].invoke({'job': "job01", "parameters": {}, "action": "run"})

                # get jenkins job info by passing job number
                resp = tools[0].invoke({'job': "job01", "number": 1, 
                                        "action": "status"})
                if not resp["inProgress"]:
                    print(resp["result"])

    """

    job: str = Field(
        description="name of the job"
    )
    action: str = Field(
        description="action of the job, like, create, run, delete"
    )
    number: int = Field(
        default=1,
        description="job number"
    )
    config_xml: str = Field(
        default='',
        description="job xml content"
    )
    parameters: dict = Field(
        default={},
        description="job parameters"
    )


class JenkinsJobRun(BaseTool):  # type: ignore[override, override]
    """Tool that execute the job"""
    name: str = "jenkins"
    description: str = (
        """A tool that is used to create, trigger and delete Jenkins jobs with,
          specified parameters."""
    )
    api_wrapper: JenkinsAPIWrapper = Field(default_factory=JenkinsAPIWrapper)  # type: ignore[arg-type]
    args_schema: Type[BaseModel] = JenkinsSchema


    def _run(
        self,
        job: str,
        action: str,
        number: Optional[int] = 1,
        config_xml: Optional[str] = "",
        parameters: Optional[dict] = {},
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> any:
        """Use the tool."""
        if action == "create":
            self.api_wrapper.create_job(
                job=job,
                config_xml=config_xml,
            )
        elif action == "run":
            return self.api_wrapper.run_job(
                job=job,
                parameters=parameters,
            )
        elif action == "delete":
            self.api_wrapper.delete_job(
                job=job,
            )
        elif action == "status":
            return self.api_wrapper.status_job(
                job=job,
                number=number
            )
        else:
            raise ValueError("'action' not matched")
