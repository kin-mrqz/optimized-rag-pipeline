import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';

import {
  DockerImageFunction,
  DockerImageCode,
  FunctionUrlAuthType,
  Architecture, 
} from 'aws-cdk-lib/aws-lambda';

import { ManagedPolicy } from 'aws-cdk-lib/aws-iam';

export class RagCdkInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Function to handle RAG API requests using custom handler
    const apiImageCode = DockerImageCode.fromImageAsset("../image", {
      cmd: ["api_app_handler.handler"],
      buildArgs: {
        platform: "linux/amd64",  // for pysqlite-binary to work with Lambda
      },
    });

    const apiFunction = new DockerImageFunction(this, "ApiFunction", {
      code: apiImageCode,
      memorySize: 256,
      timeout: cdk.Duration.seconds(30),
      architecture: Architecture.X86_64,
    });

    const functionUrl = apiFunction.addFunctionUrl({
      authType: FunctionUrlAuthType.NONE,
    });

    apiFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockFullAccess")
    );

    new cdk.CfnOutput(this, "FunctionUrl", {
      value: functionUrl.url,
    });

  }
}
