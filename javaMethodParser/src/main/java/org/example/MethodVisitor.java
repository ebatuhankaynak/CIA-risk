package org.example;

import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.ConditionalExpr;
import com.github.javaparser.ast.stmt.*;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.List;

public class MethodVisitor extends VoidVisitorAdapter<Void> {
    private int cyclomaticComplexity = 1; // Start with 1 for the method entry point
    @Override
    public void visit(IfStmt ifStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("IfStmt");
        super.visit(ifStmt, arg);

        if (ifStmt.getElseStmt().isPresent()) {
            System.out.println("ElseStmt");
            cyclomaticComplexity++;
        }
    }

    @Override
    public void visit(WhileStmt whileStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("WhileStmt");
        super.visit(whileStmt, arg);
    }

    @Override
    public void visit(ForStmt forStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("ForStmt");
        super.visit(forStmt, arg);
    }

    @Override
    public void visit(SwitchStmt switchStmt, Void arg) {
        List<SwitchEntry> switchEntries = switchStmt.getEntries();
        System.out.println("SwitchStmt");
        cyclomaticComplexity += switchEntries.size();

        for (SwitchEntry entry : switchEntries) {
            super.visit(entry, arg);
        }
    }

    @Override
    public void visit(TryStmt tryStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("TryStmt");
        NodeList<CatchClause> catchClauses = tryStmt.getCatchClauses();
        for (CatchClause catchClause : catchClauses) {
            cyclomaticComplexity++;
            super.visit(catchClause, arg);
        }

        if (tryStmt.getFinallyBlock().isPresent()) {
            cyclomaticComplexity++;
            super.visit(tryStmt.getFinallyBlock().get(), arg);
        }
    }

    @Override
    public void visit(BinaryExpr binaryExpr, Void arg) {
        if (binaryExpr.getOperator().asString().equals("&&") || binaryExpr.getOperator().asString().equals("||")) {
            System.out.println("BinaryExpr");
            cyclomaticComplexity++;
        }
        super.visit(binaryExpr, arg);
    }

    @Override
    public void visit(ConditionalExpr conditionalExpr, Void arg) {
        cyclomaticComplexity++;
        System.out.println("ConditionalExpr");
        super.visit(conditionalExpr, arg);
    }

    @Override
    public void visit(ForEachStmt forEachStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("ForEachStmt");
        super.visit(forEachStmt, arg);
    }

    @Override
    public void visit(ReturnStmt returnStmt, Void arg) {
        MethodDeclaration methodDeclaration = returnStmt.findAncestor(MethodDeclaration.class).orElse(null);

        if (methodDeclaration != null) {
            if (methodDeclaration.getBody().isPresent()) {
                NodeList<Statement> methodStatements = methodDeclaration.getBody().get().getStatements();
                if (!methodStatements.isEmpty() && (methodStatements.getLast().isPresent() && !returnStmt.equals(methodStatements.getLast().get()))) {
                    System.out.println("ReturnStmt");
                    cyclomaticComplexity++;
                }
            }
        }

        super.visit(returnStmt, arg);
    }

    @Override
    public void visit(DoStmt doStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("DoStmt");
        super.visit(doStmt, arg);
    }

    @Override
    public void visit(BreakStmt breakStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("BreakStmt");
        super.visit(breakStmt, arg);
    }

    @Override
    public void visit(ContinueStmt continueStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("ContinueStmt");
        super.visit(continueStmt, arg);
    }

    @Override
    public void visit(ThrowStmt throwStmt, Void arg) {
        cyclomaticComplexity++;
        System.out.println("ThrowStmt");
        super.visit(throwStmt, arg);
    }

    int getCyclomaticComplexity() {
        return cyclomaticComplexity;
    }
}