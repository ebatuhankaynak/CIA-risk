package org.example;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.eclipse.jgit.api.Git;
import org.eclipse.jgit.api.errors.GitAPIException;
import org.eclipse.jgit.diff.DiffEntry;
import org.eclipse.jgit.diff.DiffFormatter;
import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;

public class Main {
    //static BufferedWriter bw;
    static Map<Integer, Integer> map = new HashMap<>();

    public static void main(String[] args) {
        String srcPath = "D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset2\\commons-cli";

        String lastCommit = "";
        int i = 0;
        try (Git git = Git.open(new File(srcPath))) {
            Repository repository = git.getRepository();
            Iterable<RevCommit> commits = git.log().all().call();

            for (RevCommit commit : commits) {
                i++;
                org.eclipse.jgit.lib.ObjectId commitId = commit.getId();
                String commitHash = commitId.getName(); // Get the hash as a string
                lastCommit = commitHash;

                // revert
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

                // Checkout to the specific commit
                git.checkout().setName(commitHash).call();
                //System.out.println("Checked out to commit: " + commitHash);

                if (commit.getParentCount() > 0) {
                    RevCommit parentCommit = commit.getParent(0);
                    List<DiffEntry> diffs = getDiffEntries(repository, parentCommit, commit);
                    Set<FileMetadata> contributionAuthorXLines = new HashSet<>();
                    contributionAuthorXLines.clear();
                    for (DiffEntry diff : diffs) {
                        if(diff.getNewPath().endsWith(".java")){
                            String pathFile = srcPath + "\\" + diff.getNewPath().replace("/", "\\");
                            System.out.println("Changed file: " + pathFile);

                            map.clear();
                            // parse the file
                            extractMethods(pathFile); //get start, end lines of methods

                            // for each start, end pairs run the git log -l
                            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                                int startLine = entry.getKey();
                                int endLine = entry.getValue();
                                String gitCommand = "git log -L " + startLine + "," + endLine + ":" + pathFile;
                                //System.out.println(gitCommand);
                                //String[] command = {gitCommand, "log", "-L", startLine + "," + endLine + ":" + pathFile};

                                ProcessBuilder processBuilder = new ProcessBuilder("git", "log", "-L", startLine + "," + endLine + ":" + pathFile);

                                // Set the working directory for the process builder
                                processBuilder.directory(new File(srcPath));

                                try {
                                    Process process = processBuilder.start();
                                    BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

                                    String line;
                                    String currentAuthor = null;
                                    int linesContributed = 0;


                                    while ((line = reader.readLine()) != null) {
                                        if (line.startsWith("commit ")) {
                                            if (currentAuthor != null) {
                                                //System.out.println("Author: " + currentAuthor + ", Lines Contributed: " + linesContributed);
                                                addOrUpdateFile(contributionAuthorXLines, new FileMetadata(pathFile, currentAuthor, linesContributed));
                                            }

                                            currentAuthor = null;
                                            linesContributed = 0;
                                        } else if (line.startsWith("Author: ")) {
                                            currentAuthor = line.substring(8).trim();
                                        } else if (line.matches("^\\+.*") || line.matches("^\\-.*")) {
                                            // Count lines added or removed within the specified range
                                            if (currentAuthor != null) {
                                                linesContributed++;
                                            }
                                        }
                                    }


                                    /*if (currentAuthor != null) {
                                        System.out.println("Author: " + currentAuthor + ", Lines Contributed: " + linesContributed);
                                    }*/

                                    int exitCode = process.waitFor();
                                    if (exitCode != 0) {
                                        System.err.println("Error executing git log -L. Exit code: " + exitCode);
                                    }
                                } catch (IOException | InterruptedException e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                    }

                    if(!contributionAuthorXLines.isEmpty()){
                        BufferedWriter bw;
                        try{
                            // Print the contribution of the authors
                            ArrayList<String> rowsToWrite = new ArrayList<>();
                            for (FileMetadata pair : contributionAuthorXLines){
                                if(pair.getAuthor().equals(commit.getAuthorIdent().getName() + " <" + commit.getAuthorIdent().getEmailAddress() + ">")){
                                    rowsToWrite.add(pair.getFileName() + "," + pair.getAuthor() + "," +  pair.getContribution() + "," + ((double)pair.getContribution()/(double)calculateTotalCont(contributionAuthorXLines, pair.getFileName())*100));

                                    if(!pair.getAuthor().equals("Gary Gregory <garydgregory@gmail.com>")){
                                        BufferedWriter bw2;
                                        bw2 = new BufferedWriter(new FileWriter("D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset2\\log\\" + "log"+ commitHash + ".txt"));
                                        // Write header
                                        bw2.write(pair.getAuthor() + "\n");
                                        bw2.flush();
                                    }
                                }
                            }
                            if(!rowsToWrite.isEmpty()){
                                // Open a .csv file
                                bw = new BufferedWriter(new FileWriter("D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset2\\DevExp\\" + commitHash + ".csv"));
                                // Write header
                                bw.write("File Directory,Author,Contribution, Percentage of Contribution\n");

                                for(String str: rowsToWrite){
                                    bw.write(str + "\n");
                                }
                                bw.flush();
                            }
                        } catch (IOException e) {
                            System.out.println("cant open/write csv");
                            e.printStackTrace();
                        }
                    }

                } else {
                    System.out.println("First commit, no parent");
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
                    .map(p -> p.toString())
                    .filter(f -> f.endsWith(fileExtension))
                    .collect(Collectors.toList());
        }
        return result;
    }

    public static void extractMethods(String javaFile) {
        CompilationUnit cu = null;
        boolean skip = false;

        // Configure symbol resolver for ParserConfiguration
        ParserConfiguration parserConfiguration = new ParserConfiguration();
        ReflectionTypeSolver reflectionTypeSolver = new ReflectionTypeSolver();
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(reflectionTypeSolver);
        parserConfiguration.setSymbolResolver(symbolSolver);

        // Apply the symbol resolver configuration to the parser
        //JavaParser.setStaticConfiguration().setSymbolResolver(symbolSolver);

        com.github.javaparser.ParseResult<CompilationUnit> parseResult = null;
        try {
            // Parse the file
            cu = StaticJavaParser.parse(new FileInputStream(javaFile));
            JavaParser javaParser = new JavaParser(parserConfiguration);
            parseResult = javaParser.parse(new FileInputStream(javaFile));

        } catch (Exception e) {
            System.out.println("Skipping");
            skip = true;
        }

        if (!skip && parseResult.isSuccessful()) {
            // Get the method declarations of the file
            cu = parseResult.getResult().get();
            List<MethodDeclaration> all = cu.findAll(MethodDeclaration.class);

            // Traverse each method declarations and get details of them
            for (MethodDeclaration methodDeclaration : all) {
                //System.out.println("***************************");
                //System.out.println("File name: " + javaFile);

                int annotationOffset = 0;
                if (!methodDeclaration.getAnnotations().isEmpty()) {
                    annotationOffset = methodDeclaration.getAnnotations().get(methodDeclaration.getAnnotations().size() - 1).getRange().get().end.line;
                }

                //System.out.println("Annotation Offset: " + annotationOffset);
                String methodName = methodDeclaration.getName().toString();

                TypeDeclaration<?> classDeclaration = (TypeDeclaration) methodDeclaration.findAncestor(TypeDeclaration.class).orElse(null);
                String methodQualifiedSignature = "";

                if (cu.getPackageDeclaration().isPresent() && classDeclaration != null) {
                    PackageDeclaration packageDeclaration = cu.getPackageDeclaration().get();
                    String packageName = packageDeclaration.getNameAsString();
                    String className = classDeclaration.getNameAsString();
                    //System.out.println("Class name: " + className);

                    StringJoiner parameterTypesJoiner = new StringJoiner(", ");
                    methodDeclaration.getParameters().forEach(parameter -> parameterTypesJoiner.add(parameter.getType().asString()));
                    String parameterTypes = parameterTypesJoiner.toString();
                    methodQualifiedSignature = packageName + "." + className + "." + methodName + "(" + parameterTypes + ")";
                    //System.out.println("Function qualified signature is " + methodQualifiedSignature);
                }

                //System.out.println("Function name is: " + methodName);
                String methodSignature = "";

                if (methodQualifiedSignature.isEmpty()) {
                    methodSignature = methodName;
                } else {
                    methodSignature = methodQualifiedSignature;
                }
                //System.out.println("Function signature is: " + methodSignature);

                int methodStartLine = 0;
                if (annotationOffset != 0) {
                    methodStartLine = annotationOffset + 1;
                } else {
                    methodStartLine = methodDeclaration.getBegin().get().line;
                }
                //System.out.println("Function starting line number is: " + methodStartLine);

                int methodEndLine = methodDeclaration.getEnd().get().line;
                //System.out.println("Function ending line number is: " + methodEndLine);

                map.put(methodStartLine, methodEndLine);

                /*MethodVisitor methodVisitor = new MethodVisitor();
                methodVisitor.visit(methodDeclaration, null);
                int cyclomaticComplexity = methodVisitor.getCyclomaticComplexity();
                System.out.println("Cyclomatic Complexity: " + cyclomaticComplexity);*/


                /*try {
                    bw.write(javaFile + "\t" + methodName + "\t" + methodSignature + "\t" + methodStartLine + "\t" + methodEndLine + "\t" +cyclomaticComplexity);
                    bw.write("\n");
                    bw.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }*/
            }
        }
    }

    private static List<DiffEntry> getDiffEntries(Repository repository, RevCommit parentCommit, RevCommit commit)
            throws IOException, GitAPIException {
        try (DiffFormatter diffFormatter = new DiffFormatter(System.out)) {
            diffFormatter.setRepository(repository);
            diffFormatter.setContext(0);
            diffFormatter.setPathFilter(null);
            org.eclipse.jgit.lib.ObjectId oldHead = parentCommit.getTree();
            ObjectId head = commit.getTree();
            return diffFormatter.scan(oldHead, head);
        }
    }

    private static void addOrUpdateFile(Set<FileMetadata> fileSet, FileMetadata file) {
        for (FileMetadata existingFile : fileSet) {
            if (existingFile.getFileName().equals(file.getFileName()) && existingFile.getAuthor().equals(file.getAuthor())) {
                existingFile.setContribution(existingFile.getContribution() + file.getContribution());
                return;
            }
        }
        fileSet.add(file);
    }
    private static int calculateTotalCont(Set<FileMetadata> fileSet, String fName){
        int result = 0;
        for (FileMetadata existingFile : fileSet) {
            if (existingFile.getFileName().equals(fName)) {
                result += existingFile.getContribution();
            }
        }
        return result;
    }
}

class FileMetadata {
    private String fileName;
    private String author;
    private int contribution;

    public FileMetadata(String fileName, String author, int contribution) {
        this.fileName = fileName;
        this.author = author;
        this.contribution = contribution;
    }

    // Getters (you can generate them automatically in many IDEs)
    public String getFileName() {
        return fileName;
    }

    public String getAuthor() {
        return author;
    }

    public int getContribution() {
        return contribution;
    }

    public void setContribution(int c){
        contribution = c;
    }

    // Overriding equals and hashCode for Set operations
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        FileMetadata that = (FileMetadata) o;
        return contribution == that.contribution &&
                fileName.equals(that.fileName) &&
                author.equals(that.author);
    }

    @Override
    public int hashCode() {
        int result = fileName.hashCode();
        result = 31 * result + author.hashCode();
        result = 31 * result + Integer.hashCode(contribution);
        return result;
    }
}
