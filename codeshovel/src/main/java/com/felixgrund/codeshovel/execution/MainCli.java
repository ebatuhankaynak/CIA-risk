package com.felixgrund.codeshovel.execution;

import com.felixgrund.codeshovel.changes.Yfilerename;
import com.felixgrund.codeshovel.changes.Ymovefromfile;
import com.felixgrund.codeshovel.changes.Ymultichange;
import com.felixgrund.codeshovel.entities.Yresult;
import com.felixgrund.codeshovel.services.RepositoryService;
import com.felixgrund.codeshovel.services.impl.CachingRepositoryService;
import com.felixgrund.codeshovel.util.Utl;
import com.felixgrund.codeshovel.wrappers.Commit;
import com.felixgrund.codeshovel.wrappers.StartEnvironment;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.json.JSONArray;
import org.json.JSONObject;

import org.apache.commons.cli.*;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.lib.Repository;

import java.io.BufferedWriter;
import java.io.File;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.util.Objects;


/**
 * This class is the main entry point for command-line based CodeShovel executions.
 *
 * We are using Apache Commons CLI for facilitating command-line handling.
 *
 * @author Felix Grund
 */
public class MainCli {

    /*
	Entry point for the command line tool. If options are supplied incorrectly, a help message like this will
	be shown (printing it here because it shows the required arguments):

	usage: java -jar <codeshovel-jar-file>
		 -filepath <arg>      (required) path to the file containing the method
		 -methodname <arg>    (required) name of the method
		 -outfile <arg>       path to the output file. Default: current working directory
		 -reponame <arg>      name of the repository. Default: last part from repopath (before /.git)
		 -repopath <arg>      (required) path to the repository (on the local file system)
		 -startcommit <arg>   hash of the commit to begin with backwards history traversal. Default: HEAD
		 -startline <arg>     (required) start line of the method
	 */

	private static BufferedWriter bw;
	private static BufferedWriter bw2;

	public static void main(String[] args) {
		try{
			// Open a .csv file
			bw = new BufferedWriter(new FileWriter("D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset\\commons-cli\\Outputs\\call-graph.csv", true));
			// Write header
			bw.write("File Directory\tFunction Name\tFunction Signature\tFunction Starting Line No\tFunction Ending Line No\tCyclomatic\tResult");
			bw.write("\n");
		} catch (IOException e) {
			System.out.println("null");
			e.printStackTrace();
		}

		try (Reader reader = new FileReader("D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset\\commons-cli\\Outputs\\JavaMethodParserCommonCli.csv");
			 CSVParser csvParser = new CSVParser(reader, CSVFormat.TDF.withHeader("File Directory", "Function Name","Function Signature","Function Starting Line No","Function Ending Line No","Cyclomatic"))) {
			boolean isHeader = true;
			for (CSVRecord csvRecord : csvParser) {
				if(isHeader){
					isHeader = false;
					continue;
				}
				Yresult yresult = null;
				try {
//					CommandLine line = parser.parse(options, args);
					String repositoryPath = "D:/_SELEN/_2022-2023/CS588/GitHub_Dataset/commons-cli/commons-cli";
					// Unix vs. Windows. Probably there is a better way to do this.
					String pathDelimiter = repositoryPath.contains("\\\\") ? "\\\\" : "/";
					// Repo paths need to reference the .git directory. We add it to the path if it's not provided.
					String gitPathEnding = pathDelimiter + ".git";
					if (!repositoryPath.endsWith(gitPathEnding)) {
						repositoryPath += gitPathEnding;
					}
					// If no repo name parameter was provided we extract if from the repo path.
					String repositoryName = "commons-cli";
					if (repositoryName == null) {
						String[] split = repositoryPath.replace(gitPathEnding, "").split(pathDelimiter);
						repositoryName = split[split.length - 1];
					}
					String filePath = csvRecord.get("File Directory").substring(66);
					filePath = filePath.replace("\\", "/");
					String functionName = csvRecord.get("Function Name");
					int functionStartLine = Integer.parseInt(csvRecord.get("Function Starting Line No"));
					// If no start commit hash was provided we use HEAD.
					String startCommitName = null;
					if (startCommitName == null) {
						startCommitName = "HEAD";
					}
					// If no output file path was provided the output file will be saved in the current directory.
					String outputFilePath = csvRecord.get("Function Name");
					if (outputFilePath == null) {
						outputFilePath = System.getProperty("user.dir") + "/" + repositoryName + "-" + functionName + "-" + functionStartLine + ".json";
					}

					// Below is the start of a CodeShovel execution as we know it.
					Repository repository = Utl.createRepository(repositoryPath);
					Git git = new Git(repository);
					RepositoryService repositoryService = new CachingRepositoryService(git, repository, repositoryName, repositoryPath);
					Commit startCommit = repositoryService.findCommitByName(startCommitName);

					StartEnvironment startEnv = new StartEnvironment(repositoryService);
					startEnv.setRepositoryPath(repositoryPath);
					startEnv.setFilePath(filePath);
					startEnv.setFunctionName(functionName);
					startEnv.setFunctionStartLine(functionStartLine);
					startEnv.setStartCommitName(startCommitName);
					startEnv.setStartCommit(startCommit);
					startEnv.setFileName(Utl.getFileName(startEnv.getFilePath()));
					startEnv.setOutputFilePath(outputFilePath);

					yresult = ShovelExecution.runSingle(startEnv, startEnv.getFilePath(), true);
				} catch (ParseException e) {
					System.out.println(e.getMessage());
				} catch (Exception e) {
					System.out.println(e.getMessage());
					e.printStackTrace();
				}

				JSONArray authorData = new JSONArray();
				if (Objects.nonNull(yresult)) {
					for (String commitId : yresult.keySet()) {
						JSONObject data = new JSONObject();
						String diff = "";

						if (yresult.get(commitId) instanceof Ymultichange) {
							diff = ((Ymultichange) yresult.get(commitId)).getChanges().get(0).toJsonObject().get("diff").toString();
						} else {
							diff = yresult.get(commitId).toJsonObject().get("diff").toString();
						}

						String authorName = yresult.get(commitId).getCommit().getAuthorName();
						String authorEmail = yresult.get(commitId).getCommit().getAuthorEmail();
						String commitYear = String.valueOf(yresult.get(commitId).getCommit().getCommitDate().getYear() + 1900);
						if (!(yresult.get(commitId) instanceof Yfilerename) && !(yresult.get(commitId) instanceof Ymovefromfile)) {
							data.put("diff", diff);
							data.put("authorName", authorName);
							data.put("authorEmail", authorEmail);
							data.put("commitYear", commitYear);

							authorData.put(data);
						}
					}
				}

				try {
					bw.write(csvRecord.get("File Directory") + "\t" + csvRecord.get("Function Name") + "\t" + csvRecord.get("Function Signature") + "\t" + csvRecord.get("Function Starting Line No") + "\t" + csvRecord.get("Function Ending Line No") + "\t" + csvRecord.get("Cyclomatic")  + "\t" + authorData);
					bw.write("\n");
					bw.flush();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}