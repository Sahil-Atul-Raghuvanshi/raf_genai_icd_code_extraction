package com.raf.icd.dto;

import java.util.List;

public class ExtractionResponse {
    private List<FileExtractionResult> results;
    private Integer totalFiles;
    private Integer totalCodes;
    private String processingTime;

    public ExtractionResponse() {
    }

    public ExtractionResponse(List<FileExtractionResult> results, Integer totalFiles, Integer totalCodes, String processingTime) {
        this.results = results;
        this.totalFiles = totalFiles;
        this.totalCodes = totalCodes;
        this.processingTime = processingTime;
    }

    public List<FileExtractionResult> getResults() {
        return results;
    }

    public void setResults(List<FileExtractionResult> results) {
        this.results = results;
    }

    public Integer getTotalFiles() {
        return totalFiles;
    }

    public void setTotalFiles(Integer totalFiles) {
        this.totalFiles = totalFiles;
    }

    public Integer getTotalCodes() {
        return totalCodes;
    }

    public void setTotalCodes(Integer totalCodes) {
        this.totalCodes = totalCodes;
    }

    public String getProcessingTime() {
        return processingTime;
    }

    public void setProcessingTime(String processingTime) {
        this.processingTime = processingTime;
    }
}
