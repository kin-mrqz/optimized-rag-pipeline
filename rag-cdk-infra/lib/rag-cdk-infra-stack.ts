import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { AttributeType, BillingMode, Table } from 'aws-cdk-lib/aws-dynamodb';
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

    // Create a DynamoDB table for storing RAG data
    const ragQueryTable = new Table(this, "RagQueryTable", {
      partitionKey: { name: "query_id", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
    });


    // Function to handle RAG API requests using custom handler
    const apiImageCode = DockerImageCode.fromImageAsset("../image", {
      cmd: ["api_app_handler.handler"],
      buildArgs: {
        platform: "linux/amd64",  // for pysqlite-binary to work with Lambda
      },
    });

    // Create a Lambda function using the Docker image
    const apiFunction = new DockerImageFunction(this, "ApiFunction", {
      code: apiImageCode,
      memorySize: 256,
      timeout: cdk.Duration.seconds(30),
      architecture: Architecture.X86_64,
      environment: {
        TABLE_NAME: ragQueryTable.tableName,
      }
    });

    // Add a function URL to the Lambda function
    const functionUrl = apiFunction.addFunctionUrl({
      authType: FunctionUrlAuthType.NONE,
    });

    // Grant the Lambda function permissions to read/write to the DynamoDB table
    ragQueryTable.grantReadWriteData(apiFunction);
    apiFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockFullAccess")
    );

    // Output the API function URL
    new cdk.CfnOutput(this, "FunctionUrl", {
      value: functionUrl.url,
    });

  }
}
