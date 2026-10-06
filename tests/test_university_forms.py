import io
import unittest
import zipfile
from kazstyle.data.university_forms import extract_docx,is_docx_package


class OfficeSourceExtractionTests(unittest.TestCase):
    def test_paragraph_order_and_inline_runs(self):
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as archive:
            archive.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Өті</w:t></w:r><w:r><w:t>ніш</w:t></w:r></w:p><w:p><w:r><w:t>Мәтін</w:t></w:r></w:p></w:body></w:document>')
        raw=output.getvalue()
        self.assertTrue(is_docx_package(raw))
        self.assertEqual(extract_docx(raw),'Өтініш\nМәтін')
        self.assertFalse(is_docx_package(b'legacy OLE wrapper'+raw))

    def test_embedded_theme_zip_does_not_make_legacy_doc_a_docx(self):
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as archive:
            archive.writestr('theme/theme/theme1.xml','<theme/>')
        self.assertFalse(is_docx_package(output.getvalue()))


if __name__=='__main__':unittest.main()
