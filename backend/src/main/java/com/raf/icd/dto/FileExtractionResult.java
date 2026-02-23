package com.raf.icd.dto;

import java.util.List;

public class FileExtractionResult {
    private String fileName;
    private List<IcdCode> icdCodes;
    private String error;

    public FileExtractionResult() {
    }

    public FileExtractionResult(String fileName, List<IcdCode> icdCodes, String error) {
        this.fileName = fileName;
        this.icdCodes = icdCodes;
        this.error = error;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public List<IcdCode> getIcdCodes() {
        return icdCodes;
    }

    public void setIcdCodes(List<IcdCode> icdCodes) {
        this.icdCodes = icdCodes;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }
}
