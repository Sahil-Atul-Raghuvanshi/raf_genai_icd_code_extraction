package com.raf.icd.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.raf.icd.dto.ExtractionResponse;
import com.raf.icd.dto.FileExtractionResult;
import com.raf.icd.dto.IcdCode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Service
public class IcdExtractionService {

    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    @Value("${python.service.url}")
    private String pythonServiceUrl;

    @Value("${python.service.extract-endpoint}")
    private String extractEndpoint;

    public IcdExtractionService() {
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)  // Force HTTP/1.1 - Uvicorn doesn't support HTTP/2 upgrade
                .connectTimeout(Duration.ofSeconds(300))
                .build();
        this.objectMapper = new ObjectMapper();
    }

    public ExtractionResponse extractIcdCodes(List<MultipartFile> files) {
        long startTime = System.currentTimeMillis();

        System.out.println("Starting ICD extraction for " + files.size() + " files");

        List<FileExtractionResult> results = new ArrayList<>();
        int totalCodes = 0;

        for (MultipartFile file : files) {
            try {
                System.out.println("Processing file: " + file.getOriginalFilename());

                FileExtractionResult result = extractFromSingleFile(file);
                results.add(result);

                if (result.getIcdCodes() != null) {
                    totalCodes += result.getIcdCodes().size();
                }

            } catch (Exception e) {
                System.err.println("Error processing file: " + file.getOriginalFilename() + " - " + e.getMessage());
                FileExtractionResult errorResult = new FileExtractionResult();
                errorResult.setFileName(file.getOriginalFilename());
                errorResult.setError("Failed to process file: " + e.getMessage());
                errorResult.setIcdCodes(new ArrayList<>());
                results.add(errorResult);
            }
        }

        long endTime = System.currentTimeMillis();
        String processingTime = String.format("%.2f seconds", (endTime - startTime) / 1000.0);

        System.out.println("Extraction completed. Total codes: " + totalCodes + ", Processing time: " + processingTime);

        return new ExtractionResponse(results, files.size(), totalCodes, processingTime);
    }

    @SuppressWarnings("unchecked")
    private FileExtractionResult extractFromSingleFile(MultipartFile file) throws Exception {
        String url = pythonServiceUrl + extractEndpoint;
        System.out.println("Calling Python service at: " + url);

        String boundary = "----WebKitFormBoundary" + System.currentTimeMillis();
        String filename = file.getOriginalFilename() != null ? file.getOriginalFilename() : "upload.bin";
        byte[] fileBytes = file.getBytes();

        System.out.println("File size: " + fileBytes.length + " bytes");
        System.out.println("Boundary: " + boundary);

        // Build RFC 7578 compliant multipart/form-data body
        StringBuilder sb = new StringBuilder();
        sb.append("--").append(boundary).append("\r\n");
        sb.append("Content-Disposition: form-data; name=\"file\"; filename=\"").append(filename).append("\"\r\n");
        sb.append("Content-Type: application/octet-stream\r\n");
        sb.append("\r\n");

        byte[] header = sb.toString().getBytes(StandardCharsets.UTF_8);
        String footerStr = "\r\n--" + boundary + "--\r\n";
        byte[] footer = footerStr.getBytes(StandardCharsets.UTF_8);

        // Combine all parts
        byte[] body = new byte[header.length + fileBytes.length + footer.length];
        System.arraycopy(header, 0, body, 0, header.length);
        System.arraycopy(fileBytes, 0, body, header.length, fileBytes.length);
        System.arraycopy(footer, 0, body, header.length + fileBytes.length, footer.length);

        System.out.println("Total multipart body size: " + body.length + " bytes");
        System.out.println("Header preview: " + new String(header, StandardCharsets.UTF_8).substring(0, Math.min(200, header.length)));

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .timeout(Duration.ofSeconds(300))
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();

        System.out.println("Sending request to FastAPI...");
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        System.out.println("Python service responded with status: " + response.statusCode());
        System.out.println("Response body: " + response.body().substring(0, Math.min(500, response.body().length())));

        if (response.statusCode() != 200) {
            throw new RuntimeException("Python service returned status " + response.statusCode() + ": " + response.body());
        }

        Map<String, Object> responseBody = objectMapper.readValue(response.body(), Map.class);

        FileExtractionResult result = new FileExtractionResult();
        result.setFileName(filename);

        List<Map<String, String>> icdCodesData = (List<Map<String, String>>) responseBody.get("icd_codes");

        if (icdCodesData != null) {
            List<IcdCode> icdCodes = new ArrayList<>();
            for (Map<String, String> codeData : icdCodesData) {
                IcdCode icdCode = new IcdCode();
                icdCode.setIcd_code(codeData.get("icd_code"));
                icdCode.setIcd_description(codeData.get("icd_description"));
                icdCode.setIs_billable(codeData.get("is_billable"));
                icdCode.setEvidence_snippet(codeData.get("evidence_snippet"));
                icdCode.setLlm_reasoning(codeData.get("llm_reasoning"));
                icdCode.setChart_date(codeData.get("chart_date"));
                icdCodes.add(icdCode);
            }
            result.setIcdCodes(icdCodes);
            System.out.println("Extracted " + icdCodes.size() + " ICD codes from " + filename);
        } else {
            result.setIcdCodes(new ArrayList<>());
        }

        return result;
    }
}
