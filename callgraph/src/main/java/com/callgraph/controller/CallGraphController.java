package com.callgraph.controller;

import com.callgraph.model.CallGraphManager;
import com.callgraph.model.MethodCallExtractor;
import com.callgraph.model.callgraph.CallGraph;
import com.callgraph.model.callgraph.CallGraphEdge;
import com.callgraph.model.callgraph.CallGraphNode;
import com.callgraph.model.general.CommandUtils;
import com.callgraph.model.general.Triple;
import com.callgraph.model.pr.ChangedFileWithPath;
import com.callgraph.model.pr.FileStatus;
import com.callgraph.model.pr.PullRequestChange;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@CrossOrigin("http://localhost:3000")
@RestController
@RequestMapping("api/v1/callgraph")
public class CallGraphController {

    @PostMapping("create-save")
    public boolean createSaveCallGraph(@RequestBody Triple<String, String, String> callgraphInfo) {
        boolean isWindows = System.getProperty("os.name").toLowerCase().startsWith("windows");
        String srcPath = callgraphInfo.getValue0();
        String branchName = callgraphInfo.getValue1();
        String projectIdentifier = callgraphInfo.getValue2();

        MethodCallExtractor extractor = new MethodCallExtractor();

        try {
            CommandUtils.runCommand(srcPath, "git checkout " + branchName, isWindows);
            CallGraph callGraph = extractor.getMethodCallRelation(true, srcPath, null);
            CallGraphManager callGraphManager = new CallGraphManager();

            callGraphManager.createCallGraph(callGraph, branchName, projectIdentifier);
            callGraphManager.close();

            return true;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return false;
        }
    }

//    @PostMapping("create")
//    public Object createCallGraph(@RequestBody String srcPath) {
//        MethodCallExtractor extractor = new MethodCallExtractor();
//
//        ProjectInfo projectInfo = extractor.getMethodCallRelation(true, srcPath.toString(), null);
//
//        DirectedWeightedPseudograph<CallGraphNode, DefaultWeightedEdge> graph =
//                new DirectedWeightedPseudograph<>(DefaultWeightedEdge.class);
//
//        Map<CallGraphNode, List<CalleeFunction>> methodCallRelation = projectInfo.getCallerCallees(); // TODO This can be empty
//        HashMap<String, BasicFunctionDefNode> definedFunctions = projectInfo.getDefinedFunctions(); // TODO This can be empty
//
//        methodCallRelation.forEach((caller, calleeList) -> {
//            if (!graph.containsVertex(caller) && !calleeList.isEmpty()) {
//                graph.addVertex(caller);
//            }
//            calleeList.forEach((calleeFunction -> {
//                if (!definedFunctions.containsKey(calleeFunction.getFunctionSignature())) {
//                    // This method is not user defined
//                    return;
//                }
//                BasicFunctionDefNode functionDefNode = definedFunctions.get(calleeFunction.getFunctionSignature());
//                CallGraphNode callee = new CallGraphNode(calleeFunction.getFunctionSignature(),
//                        calleeFunction.getClassName(),
//                        calleeFunction.getFunctionName(),
//                        calleeFunction.getPackageName(),
//                        functionDefNode.getFilePath(),
//                        functionDefNode.getDeclarationStart(),
//                        functionDefNode.getDeclarationEnd());
//
//                if (!graph.containsVertex(callee)) {
//                    graph.addVertex(callee);
//                }
//                DefaultWeightedEdge edge = graph.addEdge(caller, callee);
//                graph.setEdgeWeight(edge, calleeFunction.getCallLine());
//            }));
//        });
//
//        return graph;
//    }

    @PostMapping("update")
    public boolean updateCallGraph(@RequestBody PullRequestChange pullRequestChange) {
        boolean isWindows = System.getProperty("os.name").toLowerCase().startsWith("windows");
        MethodCallExtractor extractor = new MethodCallExtractor();

        ArrayList<String> oldFilesPath = new ArrayList<>();
        ArrayList<String> newFilesPath = new ArrayList<>();

        for (ChangedFileWithPath changedFileWithPath: pullRequestChange.getChangedFilesWithPath()) {
            String filePath = changedFileWithPath.getFilePath();
            FileStatus status = changedFileWithPath.getStatus();
            if (status == FileStatus.ADDED) {
                newFilesPath.add(filePath);
            } else if (status == FileStatus.REMOVED) {
                oldFilesPath.add(filePath);
            } else if (status == FileStatus.MODIFIED) {
                newFilesPath.add(filePath);
                oldFilesPath.add(filePath);
            }
        }

        // Create mini-callgraph for the old versions of the changed files
        CommandUtils.runCommand(pullRequestChange.getSrcPath(), "git checkout " + pullRequestChange.getDestinationBranchSha(), isWindows);
        CallGraph oldCallGraph = extractor.getMethodCallRelation(false, pullRequestChange.getSrcPath(), oldFilesPath);

        // Create mini-callgraph for the new versions of the changed files
        CommandUtils.runCommand(pullRequestChange.getSrcPath(), "git checkout " + pullRequestChange.getOriginBranchSha(), isWindows);
        CallGraph newCallGraph = extractor.getMethodCallRelation(false, pullRequestChange.getSrcPath(), newFilesPath);

        // Compare the created mini-callgraphs and find added/removed function calls and declarations
        ArrayList<CallGraphNode> oldNodes = oldCallGraph.getNodeList();
        ArrayList<CallGraphEdge> oldEdges = oldCallGraph.getEdgeList();
        ArrayList<CallGraphNode> newNodes = newCallGraph.getNodeList();
        ArrayList<CallGraphEdge> newEdges = newCallGraph.getEdgeList();

        // Find the differences between two mini-callgraphs
        List<CallGraphNode> removedNodes = oldNodes.stream().filter(x -> !newNodes.contains(x)).distinct().collect(Collectors.toList());
        List<CallGraphNode> addedNodes = newNodes.stream().filter(x -> !oldNodes.contains(x)).distinct().collect(Collectors.toList());
        List<CallGraphEdge> removedEdges = oldEdges.stream().filter(x -> !newEdges.contains(x)).distinct().collect(Collectors.toList());
        List<CallGraphEdge> addedEdges = newEdges.stream().filter(x -> !oldEdges.contains(x)).distinct().collect(Collectors.toList());

        try {
            CallGraphManager callGraphManager = new CallGraphManager();
            callGraphManager.updateCallGraph(removedNodes, addedNodes, removedEdges, addedEdges, pullRequestChange.getProjectIdentifier(), pullRequestChange.getDestinationBranchName());
            callGraphManager.close();

            return true;
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return false;
        }
    }
}
