"""
Document processor module for PDF text extraction and validation.
Handles PDF file validation, text extraction, and text cleaning.
"""

import io
import re
from typing import Optional, Tuple
import PyPDF2
import streamlit as st


def validate_pdf(pdf_file) -> Tuple[bool, str]:
    """
    Validate that the uploaded file is a valid PDF and within size limits.
    
    Args:
        pdf_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if pdf_file is None:
            return False, "No file provided"
        
        # Check file extension
        if not pdf_file.name.lower().endswith('.pdf'):
            return False, "File must be a PDF format (.pdf extension required)"
        
        # Check file size (50MB limit as per design)
        max_size = 50 * 1024 * 1024  # 50MB in bytes
        if pdf_file.size > max_size:
            return False, f"File size ({pdf_file.size / (1024*1024):.1f}MB) exceeds the 50MB limit. Please use a smaller file."
        
        # Check minimum file size (avoid empty files)
        if pdf_file.size < 100:  # Less than 100 bytes is likely not a valid PDF
            return False, "File is too small to be a valid PDF document"
        
        # Try to read the PDF to validate format
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Check if PDF has pages
            if len(pdf_reader.pages) == 0:
                return False, "PDF file appears to be empty (no pages found)"
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                return False, "Encrypted PDF files are not supported. Please provide an unencrypted PDF."
            
            # Try to access the first page to ensure it's readable
            try:
                first_page = pdf_reader.pages[0]
                # Attempt to extract some text to verify readability
                test_text = first_page.extract_text()
                # Note: test_text might be empty for image-only PDFs, which is handled later
            except Exception as page_error:
                return False, f"Cannot read PDF pages: {str(page_error)}"
            
            # Reset file pointer for future use
            pdf_file.seek(0)
            
            return True, ""
            
        except PyPDF2.errors.PdfReadError as pdf_error:
            return False, f"Invalid or corrupted PDF file: {str(pdf_error)}"
        except MemoryError:
            return False, "PDF file is too large to process in available memory. Please use a smaller file."
        except Exception as read_error:
            return False, f"Error reading PDF file: {str(read_error)}"
        
    except AttributeError as attr_error:
        return False, f"Invalid file object: {str(attr_error)}"
    except Exception as e:
        return False, f"Unexpected error during PDF validation: {str(e)}"


def extract_text_from_pdf(pdf_file) -> Tuple[str, Optional[str]]:
    """
    Extract text content from all pages of a PDF file.
    
    Args:
        pdf_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (extracted_text, error_message)
    """
    try:
        # Reset file pointer
        pdf_file.seek(0)
        
        # Create PDF reader
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = ""
        total_pages = len(pdf_reader.pages)
        
        # Extract text from each page
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += f"\n--- Page {page_num + 1} ---\n"
                    extracted_text += page_text
                    extracted_text += "\n"
                    
                # Update progress if in Streamlit context
                if hasattr(st, 'session_state'):
                    progress = (page_num + 1) / total_pages
                    if 'extraction_progress' in st.session_state:
                        st.session_state.extraction_progress = progress
                        
            except Exception as e:
                # Continue with other pages if one page fails
                st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                continue
        
        if not extracted_text.strip():
            return "", "No text could be extracted from the PDF"
        
        return extracted_text, None
        
    except PyPDF2.errors.PdfReadError as e:
        return "", f"PDF reading error: {str(e)}"
    except Exception as e:
        return "", f"Error extracting text: {str(e)}"


def clean_text(raw_text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and formatting issues.
    
    Args:
        raw_text: Raw text extracted from PDF
        
    Returns:
        Cleaned text string
    """
    if not raw_text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', raw_text)  # Multiple empty lines to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace on lines
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing whitespace on lines
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\S\n]+', ' ', text)  # Non-breaking spaces and other whitespace
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Remove hyphenation across lines
    
    # Clean up page markers (but keep them for reference)
    text = re.sub(r'\n--- Page \d+ ---\n', '\n\n--- Page Break ---\n\n', text)
    
    # Remove excessive spaces around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    # Ensure proper paragraph spacing
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()


def process_pdf(pdf_file) -> Tuple[str, Optional[str]]:
    """
    Complete PDF processing pipeline: validate, extract, and clean text.
    
    Args:
        pdf_file: Streamlit uploaded file object
        
    Returns:
        Tuple of (processed_text, error_message)
    """
    # Step 1: Validate PDF
    is_valid, validation_error = validate_pdf(pdf_file)
    if not is_valid:
        return "", validation_error
    
    # Step 2: Extract text
    raw_text, extraction_error = extract_text_from_pdf(pdf_file)
    if extraction_error:
        return "", extraction_error
    
    # Step 3: Clean text
    cleaned_text = clean_text(raw_text)
    
    if not cleaned_text:
        return "", "No readable text found in the PDF after processing"
    
    return cleaned_text, None