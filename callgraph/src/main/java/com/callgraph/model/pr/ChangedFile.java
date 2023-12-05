package com.callgraph.model.pr;

public class ChangedFile {
    private String fileName;
    private FileStatus status;

    public ChangedFile() {
    }

    public ChangedFile(String fileName, FileStatus status) {
        this.fileName = fileName;
        this.status = status;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public FileStatus getStatus() {
        return status;
    }

    public void setStatus(FileStatus status) {
        this.status = status;
    }
}
