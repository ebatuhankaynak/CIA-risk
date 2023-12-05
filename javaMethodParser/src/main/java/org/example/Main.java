package org.example;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.TypeDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.Utils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.Annotation;
import java.util.List;
import java.util.StringJoiner;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class Main {
    static BufferedWriter bw;

    public static void main(String[] args) {
        try{
            // Open a .csv file
            bw = new BufferedWriter(new FileWriter(new File("D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset\\commons-cli\\commons-cli\\JavaMethodParserCommonCli.csv")));

            // Write header
            bw.write("File Directory\tFunction Name\tFunction Signature\tFunction Starting Line No\tFunction Ending Line No\tCyclomatic");
            bw.write("\n");

            bw.flush();
        } catch (IOException e) {
            System.out.println("null");
            e.printStackTrace();
        }

        try {
            // Find all java files in the directory
            List<String> files = findFiles(Paths.get("D:\\_SELEN\\_2022-2023\\CS588\\GitHub_Dataset\\commons-cli\\commons-cli"), "java");

            // For all java files, find methods
            files.forEach(x -> extractMethods(x));
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
                System.out.println("***************************");
                System.out.println("File name: " + javaFile);

                int annotationOffset = 0;
                if (!methodDeclaration.getAnnotations().isEmpty()) {
                    annotationOffset = methodDeclaration.getAnnotations().get(methodDeclaration.getAnnotations().size() - 1).getRange().get().end.line;
                }

                System.out.println("Annotation Offset: " + annotationOffset);
                String methodName = methodDeclaration.getName().toString();

                TypeDeclaration<?> classDeclaration = (TypeDeclaration) methodDeclaration.findAncestor(TypeDeclaration.class).orElse(null);
                String methodQualifiedSignature = "";

                if (cu.getPackageDeclaration().isPresent() && classDeclaration != null) {
                    PackageDeclaration packageDeclaration = cu.getPackageDeclaration().get();
                    String packageName = packageDeclaration.getNameAsString();
                    String className = classDeclaration.getNameAsString();
                    System.out.println("Class name: " + className);

                    StringJoiner parameterTypesJoiner = new StringJoiner(", ");
                    methodDeclaration.getParameters().forEach(parameter -> parameterTypesJoiner.add(parameter.getType().asString()));
                    String parameterTypes = parameterTypesJoiner.toString();
                    methodQualifiedSignature = packageName + "." + className + "." + methodName + "(" + parameterTypes + ")";
                    System.out.println("Function qualified signature is " + methodQualifiedSignature);
                }

                System.out.println("Function name is: " + methodName);
                String methodSignature = "";

                if (methodQualifiedSignature.isEmpty()) {
                    methodSignature = methodName;
                } else {
                    methodSignature = methodQualifiedSignature;
                }
                System.out.println("Function signature is: " + methodSignature);

                int methodStartLine = 0;
                if (annotationOffset != 0) {
                    methodStartLine = annotationOffset + 1;
                } else {
                    methodStartLine = methodDeclaration.getBegin().get().line;
                }
                System.out.println("Function starting line number is: " + methodStartLine);

                int methodEndLine = methodDeclaration.getEnd().get().line;
                System.out.println("Function ending line number is: " + methodEndLine);

                MethodVisitor methodVisitor = new MethodVisitor();
                methodVisitor.visit(methodDeclaration, null);
                int cyclomaticComplexity = methodVisitor.getCyclomaticComplexity();
                System.out.println("Cyclomatic Complexity: " + cyclomaticComplexity);

                try {
                    bw.write(javaFile + "\t" + methodName + "\t" + methodSignature + "\t" + methodStartLine + "\t" + methodEndLine + "\t" +cyclomaticComplexity);
                    bw.write("\n");
                    bw.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
