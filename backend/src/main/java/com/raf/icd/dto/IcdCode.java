package com.raf.icd.dto;

public class IcdCode {
    private String icd_code;
    private String icd_description;
    private String is_billable;
    private String evidence_snippet;
    private String llm_reasoning;
    private String chart_date;

    public IcdCode() {
    }

    public IcdCode(String icd_code, String icd_description, String is_billable, String evidence_snippet, String llm_reasoning, String chart_date) {
        this.icd_code = icd_code;
        this.icd_description = icd_description;
        this.is_billable = is_billable;
        this.evidence_snippet = evidence_snippet;
        this.llm_reasoning = llm_reasoning;
        this.chart_date = chart_date;
    }

    public String getIcd_code() {
        return icd_code;
    }

    public void setIcd_code(String icd_code) {
        this.icd_code = icd_code;
    }

    public String getIcd_description() {
        return icd_description;
    }

    public void setIcd_description(String icd_description) {
        this.icd_description = icd_description;
    }

    public String getIs_billable() {
        return is_billable;
    }

    public void setIs_billable(String is_billable) {
        this.is_billable = is_billable;
    }

    public String getEvidence_snippet() {
        return evidence_snippet;
    }

    public void setEvidence_snippet(String evidence_snippet) {
        this.evidence_snippet = evidence_snippet;
    }

    public String getLlm_reasoning() {
        return llm_reasoning;
    }

    public void setLlm_reasoning(String llm_reasoning) {
        this.llm_reasoning = llm_reasoning;
    }

    public String getChart_date() {
        return chart_date;
    }

    public void setChart_date(String chart_date) {
        this.chart_date = chart_date;
    }
}
