package com.callgraph.model.pr;

public class ChangedFileWithPath extends ChangedFile {
    private String filePath;

    public ChangedFileWithPath() {
    }

    public ChangedFileWithPath(String fileName, FileStatus status, String filePath) {
        super(fileName, status);
        this.filePath = filePath;
    }

    public String getFilePath() {
        return filePath;
    }

    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }
}
