package com.callgraph;

import com.callgraph.model.MethodCallExtractor;
import com.callgraph.model.SymbolSolverFactory;
import com.callgraph.model.callgraph.CallGraph;
import com.callgraph.model.callgraph.CallGraphEdge;
import com.callgraph.model.general.Utils;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import org.apache.commons.lang3.ObjectUtils;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SpringBootTest
class CallgraphServerApplicationTests {
	String srcPath = "D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset\\commons-cli\\commons-cli";

	@Test
	void getCallerandCalleeOfCommits(){

        int i = 0;
        String lastCommit = "";
		try (Git git = Git.open(new File(srcPath))) {
			Repository repository = git.getRepository();
			Iterable<RevCommit> commits = git.log().all().call();

			for (RevCommit commit : commits) {
                i++;
				if(i >= 1014) {
					ObjectId commitId = commit.getId();
					String commitHash = commitId.getName(); // Get the hash as a string
					lastCommit = commitHash;
					// Checkout to the specific commit
					git.checkout().setName(commitHash).call();
					//System.out.println("Checked out to commit: " + commitHash);

					// Create and write content to the CSV file for each commit
					getCallerandCalleeInProject(commitHash);

					try {
						File directory = new File(srcPath);
						if (!directory.exists() || !directory.isDirectory()) {
							System.out.println("Directory doesn't exist or is not a directory.");
							return;
						}

						ProcessBuilder processBuilder = new ProcessBuilder("git", "reset", "--hard", "HEAD");
						processBuilder.directory(directory);
						processBuilder.redirectErrorStream(true);

						Process process = processBuilder.start();
						int exitCode = process.waitFor();

						if (exitCode == 0) {
							//System.out.println("All changes in the directory reverted successfully.");
						} else {
							//System.out.println("Failed to revert changes in the directory. Exit code: " + exitCode);
						}
						System.out.println("Finished commit: " + commitHash);
					} catch (IOException | InterruptedException e) {
						System.out.println("****************IOException or InterruptedException");
						System.out.println(i);
						System.out.println(lastCommit);
						e.printStackTrace();
					}
				}
			}
		} catch (Exception e) {
            System.out.println("****************Exception");
            System.out.println(i);
            System.out.println(lastCommit);
			e.printStackTrace();
		}
        System.out.println("****************Result");
        System.out.println(i);
	}
	@Test
	void getCallerandCalleeInProject(String commitHash) {
		ArrayList<CallGraphEdge> edgeList = null;
		try {
			MethodCallExtractor extractor = new MethodCallExtractor();
			CallGraph callGraph = extractor.getMethodCallRelation(true, srcPath, null);
			edgeList = callGraph.getEdgeList();
		}
		catch(Exception e){
			System.out.println("**************Call Graph Exception");
			System.out.println(commitHash);
		}

		if(edgeList != null){
			try (BufferedWriter writer = new BufferedWriter(new FileWriter("D:\\_SELEN\\_2022-2023\\CS588\\commons-cli-callGraphOutput\\" + commitHash + ".csv"))) {
				// Write header
				writer.write("Caller,Callee\n");

				// Write edges data
				for (CallGraphEdge edge : edgeList) {
					writer.write(edge.getStartNodeSignature() + "," + edge.getEndNodeSignature() + "\n");
				}

				//System.out.println("CSV file has been created successfully!");
			} catch (Exception e) {
				System.out.println("****************Exception Call Graph");
				System.out.println(commitHash);
				e.printStackTrace();
			}
		}
	}

	static BufferedWriter bw;
	@Test
	void getMethodsInProject(){
		try{
			// Open a .csv file
			bw = new BufferedWriter(new FileWriter(new File(srcPath + "\\commons-cli-callgraph.csv")));
			// Write header
			bw.write("Function Signature\tFunction Starting Line No\tFunction Ending Line No");
			bw.write("\n");
		} catch (IOException e) {
			System.out.println("null");
			e.printStackTrace();
		}

		try {
			// Find all java files in the directory
			List<String> files = findFiles(Paths.get(srcPath), "java");

			// Set up parser configuration
			JavaSymbolSolver symbolSolver = SymbolSolverFactory.getJavaSymbolSolver(srcPath);
			StaticJavaParser.getParserConfiguration().setSymbolResolver(symbolSolver);
			StaticJavaParser.getParserConfiguration().setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_17);
			StaticJavaParser.getParserConfiguration().setPreprocessUnicodeEscapes(true);

			// For all java files, find methods
			files.forEach(x -> extractMethodsInAFile(x));
			bw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static List<String> findFiles(Path path, String fileExtension)
			throws IOException {

		if (!Files.isDirectory(path)) {
			throw new IllegalArgumentException("Path must be a directory!");
		}

		List<String> result;

		try (Stream<Path> walk = Files.walk(path)) {
			result = walk
					.filter(p -> !Files.isDirectory(p))
					// this is a path, not string,
					// this only test if path end with a certain path
					//.filter(p -> p.endsWith(fileExtension))
					// convert path to string first
					.map(p -> p.toString().toLowerCase())
					.filter(f -> f.endsWith(fileExtension))
					.collect(Collectors.toList());
		}
		return result;
	}
	int counter = 1;
    void extractMethodsInAFile(String javaFile) {
		CompilationUnit cu = null;
        try {
            // Parse the file
            cu = StaticJavaParser.parse(new FileInputStream(javaFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // Get the method declarations of the file
        List<MethodDeclaration> all = cu.findAll(MethodDeclaration.class);
        // Traverse each method declarations and get details of them
        for (MethodDeclaration methodDeclaration : all) {
            System.out.println("***************************");
            String methodName = methodDeclaration.getName().toString();
            System.out.println("Function name is: " + methodName);
			String functionSignature = "";
			String filePath = javaFile.substring(srcPath.length() + 1);
			try {
				functionSignature = Utils.simplifySignature(methodDeclaration.resolve().getQualifiedSignature(), false);
			} catch (Exception e) {
				System.out.println("Exception");
				functionSignature = Utils.simplifySignature(methodDeclaration.getSignature().asString(), false);
				functionSignature = filePath + "::"+ functionSignature;
			}
            System.out.println("Function signature is: " + functionSignature);
            int methodStartLine = methodDeclaration.getBegin().get().line;
            System.out.println("Function starting line number is: " + methodStartLine);
            int methodEndLine = methodDeclaration.getEnd().get().line;
            System.out.println("Function ending line number is: " + methodEndLine);

			try{
				System.out.println("Counter: " + counter++);
                bw.write(functionSignature + "\t" + methodStartLine + "\t" + methodEndLine);
                bw.write("\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
