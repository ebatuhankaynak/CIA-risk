package com.callgraph.model;

import com.callgraph.model.callgraph.CallGraph;
import com.callgraph.model.callgraph.CallGraphEdge;
import com.callgraph.model.callgraph.CallGraphNode;
import com.callgraph.model.general.Utils;
import com.github.javaparser.ParserConfiguration;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.javaparsermodel.declarations.JavaParserMethodDeclaration;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class MethodCallExtractor {
    public MethodCallExtractor() {}

    public CallGraph getMethodCallRelation(boolean isRoot, String rootPath, List<String> filePaths) {
        JavaSymbolSolver symbolSolver = SymbolSolverFactory.getJavaSymbolSolver(rootPath);
        StaticJavaParser.getParserConfiguration().setSymbolResolver(symbolSolver);
        StaticJavaParser.getParserConfiguration().setLanguageLevel(ParserConfiguration.LanguageLevel.JAVA_17);
        StaticJavaParser.getParserConfiguration().setPreprocessUnicodeEscapes(true);

        HashMap<String, CallGraphNode> nodes = new HashMap<>();
        ArrayList<CallGraphEdge> edges = new ArrayList<>();

        // If it is root, create a callgraph starting from rootPath. If it is not root, create a callgraph using given filePaths
        List<String> files = isRoot ? Utils.getFilesBySuffixInPaths("java", rootPath) : filePaths;
        for (String javaFile: files) {
            extract(rootPath, javaFile, nodes, edges);
        }

        return new CallGraph(nodes, edges);
    }


    private void extract(String srcPath, String javaFile, HashMap<String, CallGraphNode> nodes, ArrayList<CallGraphEdge> edges) {
        CompilationUnit cu = null;
        try {
            cu = StaticJavaParser.parse(new FileInputStream(javaFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // Get the method declaration and traverse it
        List<MethodDeclaration> all = cu.findAll(MethodDeclaration.class);
        for (MethodDeclaration methodDeclaration : all) {
            ArrayList<String> calleeFunctions = new ArrayList<>();
            CallGraphNode callGraphNode = null;

            // Iterate over the contents of each method declaration to find other methods that are called internally
            methodDeclaration.accept(new MethodCallVisitor(), calleeFunctions);
            String functionSignature = "";
            String functionName = "";
            String className = "";
            String packageName = "";
            String filePath = javaFile.substring(srcPath.length() + 1);
            try {
                functionSignature = Utils.simplifySignature(methodDeclaration.resolve().getQualifiedSignature(), false);
                functionName = methodDeclaration.resolve().getName();
                className = methodDeclaration.resolve().getClassName();
                packageName = methodDeclaration.resolve().getPackageName();
                //System.out.println("function signature: " + functionSignature);
            } catch (Exception e) {
                functionSignature = filePath + "::"+ Utils.simplifySignature(methodDeclaration.getSignature().asString(), false);
                functionName = methodDeclaration.getName().toString();
                className = methodDeclaration.getClass().toString();
                //System.out.println("exception-function signature: " + functionSignature);
            }
            System.out.println("function signature: " + functionSignature);
            System.out.println("************************");

            //assert functionSignature != null;
            /*try(FileWriter fw = new FileWriter("myfile.txt", true);
                BufferedWriter bw = new BufferedWriter(fw);
                PrintWriter out = new PrintWriter(bw))
            {
                out.println("**************************");
                out.println("class name: " + className);
                out.println("function name: " + functionName);
                out.println("package name: " + packageName);
                out.println("function signature: " + functionSignature);
                out.println("file path: " + filePath);
                out.println("callee functions: " + calleeFunctions);
            } catch (IOException e) {
                //exception handling left as an exercise for the reader
            }*/

            callGraphNode = new CallGraphNode(className, functionName, packageName, functionSignature, filePath);

            nodes.put(functionSignature, callGraphNode);
            for (String calleeSignature: calleeFunctions) {
                edges.add(new CallGraphEdge(functionSignature, calleeSignature));
            }

            // Remove possible duplicates
            HashSet<CallGraphEdge> edgeSet = new HashSet<>(edges);
            edges.clear();
            edges.addAll(edgeSet);
        }
    }

    private static class MethodCallVisitor extends VoidVisitorAdapter<List<String>> {
        public MethodCallVisitor() {}

        public void visit(MethodCallExpr n, List<String> calleeFunctions) {
            ResolvedMethodDeclaration resolvedMethodDeclaration = null;
            try {
                resolvedMethodDeclaration = n.resolve();
                String signature = Utils.simplifySignature(resolvedMethodDeclaration.getQualifiedSignature(), false);
                if (resolvedMethodDeclaration instanceof JavaParserMethodDeclaration) {
                    calleeFunctions.add(signature);
                }
            } catch (Exception e) {
                String signature = n.getName().asString();
                System.out.println("Exception- " + signature);
                System.out.println("Type arguments: "+ n.getTypeArguments());
                System.out.println("Scope: " + n.getScope().toString());
                System.out.println("Arguments: " + n.getArguments());

                /*ResolvedType resolvedType = n.getScope().calculateResolvedType();
                System.out.println(resolvedType.describe());*/

//                logger.error("Line {}, {} cannot resolve some symbol, because {}",
//                        n.getRange().get().begin.line,
//                        n.getNameAsString() + n.getArguments().toString().replace("[", "(").replace("]", ")"),
//                        e.getMessage());
            }
            // Don't forget to call super, it may find more method calls inside the arguments of this method call, for example.
            super.visit(n, calleeFunctions);
        }
    }
}
