package com.callgraph.model;

import com.callgraph.model.general.Utils;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JarTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.symbolsolver.utils.SymbolSolverCollectionStrategy;
import com.github.javaparser.utils.ProjectRoot;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SymbolSolverFactory {
    public static JavaSymbolSolver getJavaSymbolSolver(String srcPaths) {
        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();

        ReflectionTypeSolver reflectionTypeSolver = new ReflectionTypeSolver();
        combinedTypeSolver.add(reflectionTypeSolver);

        ProjectRoot projectRoot = new SymbolSolverCollectionStrategy().collect(new File(srcPaths).toPath());
        projectRoot.getSourceRoots().forEach(root -> combinedTypeSolver.add(new JavaParserTypeSolver(root.getRoot())));

        List<JarTypeSolver> jarTypeSolvers = makeJarTypeSolvers(srcPaths);
        jarTypeSolvers.forEach(combinedTypeSolver::add);

        return new JavaSymbolSolver(combinedTypeSolver);
    }

    private static List<JarTypeSolver> makeJarTypeSolvers(String libPaths) {
        List<String> jarPaths = Utils.getFilesBySuffixInPaths("jar", libPaths);
        List<JarTypeSolver> jarTypeSolvers = new ArrayList<>(jarPaths.size());
        try {
            for (String jarPath : jarPaths) {
                jarTypeSolvers.add(new JarTypeSolver(jarPath));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return jarTypeSolvers;
    }
}
