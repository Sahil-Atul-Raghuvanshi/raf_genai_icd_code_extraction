import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [servicesReady, setServicesReady] = useState(false);
  const [checkingServices, setCheckingServices] = useState(true);

  // Check if all services are running
  useEffect(() => {
    const checkServices = async () => {
      try {
        setCheckingServices(true);
        
        // Check backend (Spring Boot)
        const backendResponse = await fetch('/api/health', {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
        }).catch(() => null);

        if (!backendResponse || !backendResponse.ok) {
          setServicesReady(false);
          setError('Backend service is not running. Please start all services using START_ALL_SERVICES.bat');
          setCheckingServices(false);
          return;
        }

        // If backend is up, assume Python service is also up (backend checks Python on extraction)
        setServicesReady(true);
        setError(null);
        setCheckingServices(false);
        
      } catch (err) {
        console.error('Service check error:', err);
        setServicesReady(false);
        setError('Unable to connect to services. Please start all services using START_ALL_SERVICES.bat');
        setCheckingServices(false);
      }
    };

    // Initial check
    checkServices();

    // Recheck every 5 seconds
    const interval = setInterval(checkServices, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = (event) => {
    const selectedFiles = Array.from(event.target.files);
    const validFiles = selectedFiles.filter(file => {
      const extension = file.name.split('.').pop().toLowerCase();
      return ['pdf', 'txt', 'doc', 'docx'].includes(extension);
    });

    if (validFiles.length !== selectedFiles.length) {
      setError('Some files were rejected. Only PDF, TXT, DOC, DOCX files are allowed.');
    }

    const newFiles = validFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file: file,
      name: file.name,
      size: (file.size / 1024).toFixed(2) + ' KB',
      type: file.name.split('.').pop().toUpperCase()
    }));

    setFiles([...files, ...newFiles]);
    event.target.value = null; // Reset input
  };

  const removeFile = (id) => {
    setFiles(files.filter(file => file.id !== id));
  };

  const handleExtractICD = async () => {
    if (files.length === 0) {
      setError('Please upload at least one file');
      return;
    }

    if (!servicesReady) {
      setError('Services are not ready. Please wait or restart services.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      files.forEach(fileObj => {
        formData.append('files', fileObj.file);
      });

      const response = await axios.post('/api/extract-icd', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (err) {
      console.error('Error extracting ICD codes:', err);
      setError(err.response?.data?.message || 'Failed to extract ICD codes. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setFiles([]);
    setResults(null);
    setError(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🏥 RAF ICD Extraction System</h1>
          <p>Upload clinical documents to extract ICD-10 codes using AI</p>
          <div className="service-status">
            {checkingServices ? (
              <span className="status-checking">
                <span className="status-dot"></span>
                Checking services...
              </span>
            ) : servicesReady ? (
              <span className="status-ready">
                <span className="status-dot"></span>
                All services running
              </span>
            ) : (
              <span className="status-not-ready">
                <span className="status-dot"></span>
                Services not available - Please run START_ALL_SERVICES.bat
              </span>
            )}
          </div>
        </header>

        {error && (
          <div className="alert alert-error">
            <span>⚠️ {error}</span>
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}

        <div className="upload-section">
          <div className="upload-box">
            <input
              type="file"
              id="file-input"
              multiple
              accept=".pdf,.txt,.doc,.docx"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <div className="upload-content">
              <div className="upload-icon">📄</div>
              <h3>Upload Clinical Documents</h3>
              <p>PDF, TXT, DOC, DOCX files supported</p>
              <label htmlFor="file-input" className="btn btn-upload">
                Choose Files
              </label>
            </div>
          </div>
        </div>

        {files.length > 0 && (
          <div className="files-section">
            <div className="files-header">
              <h2>Uploaded Documents ({files.length})</h2>
              <button onClick={clearAll} className="btn btn-secondary">
                Clear All
              </button>
            </div>
            <div className="files-grid">
              {files.map(file => (
                <div key={file.id} className="file-card">
                  <div className="file-icon">
                    {file.type === 'PDF' && '📕'}
                    {file.type === 'TXT' && '📝'}
                    {(file.type === 'DOC' || file.type === 'DOCX') && '📘'}
                  </div>
                  <div className="file-info">
                    <h4>{file.name}</h4>
                    <p>{file.size} • {file.type}</p>
                  </div>
                  <button
                    className="btn-remove"
                    onClick={() => removeFile(file.id)}
                    title="Remove file"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
            <button
              onClick={handleExtractICD}
              disabled={loading || !servicesReady || checkingServices}
              className="btn btn-primary btn-extract"
              title={!servicesReady ? 'Services are not running. Please start all services.' : ''}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Processing...
                </>
              ) : checkingServices ? (
                <>
                  <span className="spinner"></span>
                  Checking Services...
                </>
              ) : !servicesReady ? (
                <>
                  ⚠️ Services Not Ready
                </>
              ) : (
                <>
                  🔍 Extract ICD-10 Codes
                </>
              )}
            </button>
          </div>
        )}

        {results && (
          <div className="results-section">
            <h2>📊 Extraction Results</h2>
            
            {results.results && results.results.map((result, idx) => (
              <div key={idx} className="result-card">
                <div className="result-header">
                  <h3>📄 {result.fileName}</h3>
                  <span className="badge">
                    {result.icdCodes?.length || 0} codes found
                  </span>
                </div>

                {result.error ? (
                  <div className="error-message">
                    <p>❌ {result.error}</p>
                  </div>
                ) : (
                  <>
                    {result.icdCodes && result.icdCodes.length > 0 ? (
                      <div className="icd-codes-grid">
                        {result.icdCodes.map((code, codeIdx) => (
                          <div key={codeIdx} className="icd-code-card">
                            <div className="icd-code-header">
                              <span className="icd-code">{code.icd_code}</span>
                              {code.is_billable === 'Yes' && (
                                <span className="badge badge-success">Billable</span>
                              )}
                              {code.chart_date && (
                                <span className="badge badge-info">{code.chart_date}</span>
                              )}
                            </div>
                            <p className="icd-description">{code.icd_description}</p>
                            
                            {code.evidence_snippet && (
                              <div className="icd-evidence">
                                <strong>📋 Evidence:</strong>
                                <p className="evidence-text">"{code.evidence_snippet}"</p>
                              </div>
                            )}
                            
                            {code.llm_reasoning && (
                              <details className="icd-reasoning">
                                <summary><strong>AI Reasoning</strong></summary>
                                <p className="reasoning-text">{code.llm_reasoning}</p>
                              </details>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="empty-state">
                        <p>No ICD-10 codes found in this document</p>
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}

            <div className="results-summary">
              <h3>Summary</h3>
              <div className="summary-stats">
                <div className="stat">
                  <span className="stat-value">{results.totalFiles || 0}</span>
                  <span className="stat-label">Documents Processed</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{results.totalCodes || 0}</span>
                  <span className="stat-label">ICD Codes Extracted</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{results.processingTime || 'N/A'}</span>
                  <span className="stat-label">Processing Time</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
