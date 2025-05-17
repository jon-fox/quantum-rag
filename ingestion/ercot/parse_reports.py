"""
ERCOT Report Parser

This module provides functionality to parse ERCOT reports (PDFs, CSVs)
and extract structured data for the RAG pipeline.
"""
import os
import io
import csv
import logging
from typing import Dict, List, Any, Optional, Union, BinaryIO
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import LangChain document loaders
try:
    from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Using traditional document loading methods.")
    LANGCHAIN_AVAILABLE = False

class ERCOTReportParser:
    """Parser for ERCOT reports in various formats"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ERCOT report parser.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.use_langchain = self.config.get("use_langchain", LANGCHAIN_AVAILABLE)
        
        # For chunking documents
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        
        # Try to load PDF libraries but make them optional
        try:
            import pdfplumber
            self.pdf_parser = pdfplumber
            self.pdf_available = True
        except ImportError:
            logger.warning("pdfplumber not installed. PDF parsing will be unavailable.")
            self.pdf_available = False
    
    def parse_report(self, file_path: str, report_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse an ERCOT report file and extract structured data.
        
        Args:
            file_path: Path to the report file
            report_type: Type of report for specialized parsing
            
        Returns:
            Dictionary of structured data extracted from the report
        """
        # Use LangChain loaders if available and enabled
        if self.use_langchain:
            try:
                return self._langchain_parse_report(file_path, report_type)
            except Exception as e:
                logger.error(f"LangChain parsing failed: {str(e)}. Falling back to traditional parsing.")
                
        # Traditional parsing as fallback
        # Determine file type from extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return self._parse_pdf(file_path, report_type)
        elif ext == '.csv':
            return self._parse_csv(file_path, report_type)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _langchain_parse_report(self, file_path: str, report_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse report using LangChain document loaders.
        
        Args:
            file_path: Path to the report file
            report_type: Type of report for specialized parsing
            
        Returns:
            Dictionary of structured data
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not available")
            
        logger.info(f"Parsing report with LangChain: {file_path}")
        
        # Choose appropriate loader based on file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        # Load documents
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Process documents based on report type if needed
        if report_type == "system_adequacy":
            return self._process_system_adequacy_documents(split_docs)
        elif report_type == "demand_forecast":
            return self._process_demand_forecast_documents(split_docs)
        
        # Generic document processing
        return {
            "documents": split_docs,
            "metadata": {
                "file_path": file_path,
                "report_type": report_type or "generic",
                "num_chunks": len(split_docs),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "processed_with": "langchain"
            }
        }
        
    def _process_system_adequacy_documents(self, documents):
        """Process system adequacy documents with specialized logic"""
        # Implement specialized processing for system adequacy reports
        return {
            "documents": documents,
            "report_type": "system_adequacy",
            "processed_with": "langchain"
        }
        
    def _process_demand_forecast_documents(self, documents):
        """Process demand forecast documents with specialized logic"""
        # Implement specialized processing for demand forecast reports
        return {
            "documents": documents,
            "report_type": "demand_forecast",
            "processed_with": "langchain"
        }
        
    def _parse_pdf(self, file_path: str, report_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse PDF report and extract data.
        
        Args:
            file_path: Path to PDF file
            report_type: Type of report for specialized parsing
            
        Returns:
            Dictionary of structured data
        """
        if not self.pdf_available:
            raise RuntimeError("PDF parsing is not available. Install pdfplumber to enable this feature.")
            
        logger.info(f"Parsing PDF report: {file_path}")
        
        # Extract text from PDF
        with self.pdf_parser.open(file_path) as pdf:
            text_content = []
            for page in pdf.pages:
                text_content.append(page.extract_text())
            
            full_text = "\n".join(text_content)
            
        # Parse based on report type
        if report_type == "system_adequacy":
            return self._parse_system_adequacy_report(full_text)
        elif report_type == "demand_forecast":
            return self._parse_demand_forecast_report(full_text)
        else:
            # Generic extraction - split into sections and extract key metrics
            return self._generic_pdf_extraction(full_text)
    
    def _parse_csv(self, file_path: str, report_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse CSV report and extract data.
        
        Args:
            file_path: Path to CSV file
            report_type: Type of report for specialized parsing
            
        Returns:
            Dictionary of structured data
        """
        logger.info(f"Parsing CSV report: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        
        result = {
            "source": file_path,
            "type": report_type or "unknown",
            "parsed_at": datetime.now().isoformat(),
            "data": data
        }
        
        # Apply specialized parsing based on report type if needed
        if report_type == "hourly_load":
            result = self._process_hourly_load_data(result)
        elif report_type == "generation_mix":
            result = self._process_generation_mix_data(result)
            
        return result
    
    def _generic_pdf_extraction(self, text: str) -> Dict[str, Any]:
        """
        Generic PDF text extraction and parsing.
        
        Args:
            text: Extracted text from PDF
            
        Returns:
            Dictionary of extracted information
        """
        # This is a simplified placeholder for actual PDF parsing logic
        # In a real implementation, this would use regex patterns or NLP techniques
        
        lines = text.split("\n")
        
        # Extract document metadata
        metadata = {}
        for i, line in enumerate(lines[:20]):  # Check first 20 lines for metadata
            if "Report Date:" in line:
                metadata["report_date"] = line.split("Report Date:")[1].strip()
            elif "Published:" in line:
                metadata["published_date"] = line.split("Published:")[1].strip()
            elif "ERCOT " in line and " Report" in line:
                metadata["title"] = line.strip()
        
        # Extract content sections
        sections = {}
        current_section = "general"
        sections[current_section] = []
        
        for line in lines:
            if line.isupper() and len(line) > 5 and len(line) < 50:
                # Likely a section header
                current_section = line.strip()
                sections[current_section] = []
            else:
                sections[current_section].append(line)
        
        # Convert section content to text
        for section, content in sections.items():
            sections[section] = "\n".join(content)
        
        return {
            "metadata": metadata,
            "sections": sections,
            "full_text": text,
            "parsed_at": datetime.now().isoformat()
        }
    
    def _parse_system_adequacy_report(self, text: str) -> Dict[str, Any]:
        """
        Parse System Adequacy Report.
        
        Args:
            text: Extracted text from PDF
            
        Returns:
            Dictionary of structured data
        """
        # Placeholder for specialized parsing logic
        # In a real implementation, this would extract specific metrics
        
        return {
            "type": "system_adequacy",
            "parsed_at": datetime.now().isoformat(),
            "data": {
                "summary": "Specialized parsing of system adequacy report would go here",
                "full_text": text[:500] + "..." if len(text) > 500 else text
            }
        }
    
    def _parse_demand_forecast_report(self, text: str) -> Dict[str, Any]:
        """
        Parse Demand Forecast Report.
        
        Args:
            text: Extracted text from PDF
            
        Returns:
            Dictionary of structured data
        """
        # Placeholder for specialized parsing logic
        # In a real implementation, this would extract specific metrics
        
        return {
            "type": "demand_forecast",
            "parsed_at": datetime.now().isoformat(),
            "data": {
                "summary": "Specialized parsing of demand forecast report would go here",
                "full_text": text[:500] + "..." if len(text) > 500 else text
            }
        }
    
    def _process_hourly_load_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process hourly load data from CSV.
        
        Args:
            data: Dictionary with parsed CSV data
            
        Returns:
            Processed data dictionary
        """
        # Placeholder for specialized processing
        data["processed"] = True
        data["summary"] = "Hourly load data processed"
        return data
    
    def _process_generation_mix_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process generation mix data from CSV.
        
        Args:
            data: Dictionary with parsed CSV data
            
        Returns:
            Processed data dictionary
        """
        # Placeholder for specialized processing
        data["processed"] = True
        data["summary"] = "Generation mix data processed"
        return data