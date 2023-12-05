package com.callgraph.model.pr;

import java.util.ArrayList;

public class PullRequestChange {
    private String srcPath;
    private String projectIdentifier;
    private String prNumber;
    private String originBranchSha;
    private String destinationBranchSha;
    private String destinationBranchName;
    private Integer impactLevel;
    private ArrayList<ChangedFileWithSha> changedFilesWithSha;
    private ArrayList<ChangedFileWithPath> changedFilesWithPath;

    public PullRequestChange() {
    }

    public PullRequestChange(String srcPath, String projectIdentifier, String prNumber, String originBranchSha, String destinationBranchSha, String destinationBranchName, Integer impactLevel, ArrayList<ChangedFileWithSha> changedFilesWithSha, ArrayList<ChangedFileWithPath> changedFilesWithPath) {
        this.srcPath = srcPath;
        this.projectIdentifier = projectIdentifier;
        this.prNumber = prNumber;
        this.originBranchSha = originBranchSha;
        this.destinationBranchSha = destinationBranchSha;
        this.destinationBranchName = destinationBranchName;
        this.impactLevel = impactLevel;
        this.changedFilesWithSha = changedFilesWithSha;
        this.changedFilesWithPath = changedFilesWithPath;
    }

    public String getSrcPath() {
        return srcPath;
    }

    public void setSrcPath(String srcPath) {
        this.srcPath = srcPath;
    }

    public String getProjectIdentifier() {
        return projectIdentifier;
    }

    public void setProjectIdentifier(String projectIdentifier) {
        this.projectIdentifier = projectIdentifier;
    }

    public String getPrNumber() {
        return prNumber;
    }

    public void setPrNumber(String prNumber) {
        this.prNumber = prNumber;
    }

    public String getOriginBranchSha() {
        return originBranchSha;
    }

    public void setOriginBranchSha(String originBranchSha) {
        this.originBranchSha = originBranchSha;
    }

    public String getDestinationBranchSha() {
        return destinationBranchSha;
    }

    public void setDestinationBranchSha(String destinationBranchSha) {
        this.destinationBranchSha = destinationBranchSha;
    }

    public String getDestinationBranchName() {
        return destinationBranchName;
    }

    public void setDestinationBranchName(String destinationBranchName) {
        this.destinationBranchName = destinationBranchName;
    }

    public Integer getImpactLevel() {
        return impactLevel;
    }

    public void setImpactLevel(Integer impactLevel) {
        this.impactLevel = impactLevel;
    }

    public ArrayList<ChangedFileWithSha> getChangedFilesWithSha() {
        return changedFilesWithSha;
    }

    public void setChangedFilesWithSha(ArrayList<ChangedFileWithSha> changedFilesWithSha) {
        this.changedFilesWithSha = changedFilesWithSha;
    }

    public ArrayList<ChangedFileWithPath> getChangedFilesWithPath() {
        return changedFilesWithPath;
    }

    public void setChangedFilesWithPath(ArrayList<ChangedFileWithPath> changedFilesWithPath) {
        this.changedFilesWithPath = changedFilesWithPath;
    }
}
