package com.raf.icd.controller;

import com.raf.icd.dto.ErrorResponse;
import com.raf.icd.dto.ExtractionResponse;
import com.raf.icd.service.IcdExtractionService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequestMapping("/api")
public class IcdExtractionController {

    private final IcdExtractionService icdExtractionService;

    public IcdExtractionController(IcdExtractionService icdExtractionService) {
        this.icdExtractionService = icdExtractionService;
    }

    @PostMapping("/extract-icd")
    public ResponseEntity<?> extractIcdCodes(@RequestParam("files") List<MultipartFile> files) {
        try {
            System.out.println("Received request to extract ICD codes from " + files.size() + " files");

            // Validate files
            if (files == null || files.isEmpty()) {
                return ResponseEntity
                        .badRequest()
                        .body(new ErrorResponse("No files provided", "VALIDATION_ERROR", 400));
            }

            // Validate file types
            for (MultipartFile file : files) {
                String filename = file.getOriginalFilename();
                if (filename == null || !isValidFileType(filename)) {
                    return ResponseEntity
                            .badRequest()
                            .body(new ErrorResponse(
                                    "Invalid file type: " + filename + ". Only PDF, TXT, DOC, DOCX are allowed",
                                    "VALIDATION_ERROR",
                                    400
                            ));
                }
            }

            // Extract ICD codes
            ExtractionResponse response = icdExtractionService.extractIcdCodes(files);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            System.err.println("Error extracting ICD codes: " + e.getMessage());
            e.printStackTrace();
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(new ErrorResponse(
                            "Failed to extract ICD codes: " + e.getMessage(),
                            "EXTRACTION_ERROR",
                            500
                    ));
        }
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Spring Boot Backend is running");
    }

    private boolean isValidFileType(String filename) {
        String lowerCase = filename.toLowerCase();
        return lowerCase.endsWith(".pdf") ||
               lowerCase.endsWith(".txt") ||
               lowerCase.endsWith(".doc") ||
               lowerCase.endsWith(".docx");
    }
}
