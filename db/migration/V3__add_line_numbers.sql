-- Add line number columns for text-based citation references.
-- Enables citations to reference both bounding boxes (for PDF highlighting)
-- and line numbers (for text display).

ALTER TABLE document_chunks ADD COLUMN line_start integer;
ALTER TABLE document_chunks ADD COLUMN line_end integer;

ALTER TABLE citations ADD COLUMN line_start integer;
ALTER TABLE citations ADD COLUMN line_end integer;

ALTER TABLE document_page_blocks ADD COLUMN line_start integer;
ALTER TABLE document_page_blocks ADD COLUMN line_end integer;
