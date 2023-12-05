package com.callgraph.model.pr;

public class ChangedFileWithSha extends ChangedFile {
    private String oldSha;
    private String newSha;

    public ChangedFileWithSha() {
    }

    public ChangedFileWithSha(String fileName, FileStatus status, String oldSha, String newSha) {
        super(fileName, status);
        this.oldSha = oldSha;
        this.newSha = newSha;
    }

    public String getOldSha() {
        return oldSha;
    }

    public void setOldSha(String oldSha) {
        this.oldSha = oldSha;
    }

    public String getNewSha() {
        return newSha;
    }

    public void setNewSha(String newSha) {
        this.newSha = newSha;
    }
}
