# Save as test_parser_quality_FULL.py

from core import init_logger
init_logger()
from services.document_parser import DocumentParser
from pathlib import Path
import json

parser = DocumentParser()

# Test ALL key fixed PDFs + show FULL content
test_files = [
    # 'hr_policies/24-25-DC-CatalogFINAL-5.17.24_compressed.pdf',
    # 'hr_policies/DineCollegeGovtRetirementPlanConversionNotiRetiremnt.pdf', 
    # 'hr_policies/PPPM - 2021 - Updated 02.23.2024 HR.pdf',
    'hr_policies/FINAL-Rev2024-DC RES LIFE HANDBOOK (7).pdf'
    # 'hr_policies/EBP_-_Prsnt_Chptr_Orient_2013 Employee Health Beneift Program.pdf'
]

results = {}
total_chars = 0

print("🔍 TESTING PARSER QUALITY - WRITING FULL CONTENT TO FILES...\n")

for rel_path in test_files:
    file_path = Path('D:/jericho/data/documents') / rel_path
    if file_path.exists():
        result = parser.parse_file(file_path)
        if result:
            # Save FULL content to file
            output_file = Path(f"parser_output_{Path(rel_path).stem}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== {rel_path} ===\n")
                f.write(f"Method: {result.extraction_method}\n")
                f.write(f"Confidence: {result.extraction_confidence}\n")
                f.write(f"Blocks: {len(result.content)}\n")
                f.write(f"Total chars: {sum(len(c) for c in result.content)}\n\n")
                
                for i, block in enumerate(result.content[:20]):  # First 20 blocks
                    f.write(f"[{i+1}] {block[:1000]}\n{'-'*80}\n")
                
                if len(result.content) > 20:
                    f.write(f"... and {len(result.content)-20} more blocks\n")
            
            chars = sum(len(c) for c in result.content)
            total_chars += chars
            
            results[rel_path] = {
                'method': result.extraction_method,
                'confidence': result.extraction_confidence,
                'blocks': len(result.content),
                'total_chars': chars,
                'output_file': str(output_file)
            }
            
            print(f"✅ {rel_path}")
            print(f"   Method: {result.extraction_method}")
            print(f"   Confidence: {result.extraction_confidence}")
            print(f"   Blocks: {len(result.content)}")
            print(f"   Chars: {chars:,}")
            print(f"   📄 Saved: {output_file}")
            print()
        else:
            print(f"❌ FAILED: {rel_path}")
    else:
        print(f"❌ MISSING: {rel_path}")

# Summary
print("="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Total files tested: {len([f for f in test_files if (Path('D:/jericho/data/documents')/f).exists()])}")
print(f"Total characters extracted: {total_chars:,}")
print("\n📁 Output files created:")
for fname, data in results.items():
    print(f"   {data['output_file']} ({data['blocks']} blocks, {data['total_chars']:,} chars)")

print("\n✅ View files with: notepad parser_output_*.txt")

