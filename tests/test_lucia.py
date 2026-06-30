import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import lucia

# Mocking external dependencies that require GPU or heavy downloads
@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch('lucia.WhisperModel'), \
         patch('lucia.Pipeline.from_pretrained'), \
         patch('lucia.sd.InputStream'), \
         patch('lucia.sd.query_devices'), \
         patch('lucia.requests.post') as mock_post:
        
        # Setup a default mock response for Ollama
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "---\ndata: 2026-06-12\nhora: 12:00\ntipo: reuniao\n---\n# Resumo da Reunião\nTeste de resumo."
        }
        mock_post.return_value = mock_response
        yield mock_post

def test_synthesize_summary_ollama_format(mock_dependencies):
    """Valida se o prompt enviado ao Ollama contém os dados de data/hora e o formato correto."""
    transcriptions = ["Olá mundo", "Teste de transcrição"]
    
    with patch('lucia.datetime') as mock_date:
        mock_date.now.return_value = datetime(2026, 6, 12, 10, 30)
        # We need to mock strftime because datetime.now() returns a real datetime object usually
        # but here we mocked the class. Let's adjust.
        
    # Re-patching specifically for the string formatting
    fixed_now = datetime(2026, 6, 12, 10, 30)
    with patch('lucia.datetime') as mock_dt:
        mock_dt.now.return_value = fixed_now
        # In the script it calls datetime.now().strftime(...)
        # fixed_now.strftime works normally.
        
        result = lucia.synthesize_summary_ollama(transcriptions)
    
    # Check if requests.post was called
    assert mock_dependencies.called
    args, kwargs = mock_dependencies.call_args
    prompt = kwargs['json']['prompt']
    
    assert "data: 2026-06-12" in prompt
    assert "hora: 10:30" in prompt
    assert "RETORNE EXATAMENTE NESTE FORMATO MARKDOWN" in prompt
    assert "[[João]]" in prompt # Check if example is in prompt

def test_summarize_existing_file(tmp_path, mock_dependencies):
    """Valida se o resumo é inserido no topo do arquivo existente."""
    test_file = tmp_path / "reuniao.md"
    original_content = "# Transcrição\nConteúdo original aqui."
    test_file.write_text(original_content, encoding="utf-8")
    
    lucia.summarize_existing_file(str(test_file))
    
    updated_content = test_file.read_text(encoding="utf-8")
    
    # Check if the summary is at the top
    assert updated_content.startswith("---")
    assert "tipo: reuniao" in updated_content
    assert "# Resumo da Reunião" in updated_content
    assert "## Transcrição Original" in updated_content
    assert original_content in updated_content

def test_summarize_already_summarized_file(tmp_path, mock_dependencies):
    """Garante que não resume duas vezes se já houver um resumo."""
    test_file = tmp_path / "ja_resumido.md"
    content = "# Resumo da Reunião\nJá tenho um resumo."
    test_file.write_text(content, encoding="utf-8")
    
    with patch('lucia.log.warning') as mock_log:
        lucia.summarize_existing_file(str(test_file))
        mock_log.assert_called_with("Este arquivo aparentemente já possui um resumo.")
    
    # Content should be unchanged
    assert test_file.read_text(encoding="utf-8") == content

def test_get_md_filename_auto(mock_dependencies):
    """Testa a geração automática do nome do arquivo."""
    with patch('lucia.input', return_value=""):
        filename = lucia.get_md_filename()
        assert filename.endswith(".md")
        assert "transcricao-" in filename

def test_process_audio_flow(tmp_path, mock_dependencies):
    """Simula o fluxo completo de processamento de áudio (Mockado)."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_text("dummy audio content")
    md_file = tmp_path / "output.md"
    
    # Mocking Faster-Whisper segments
    mock_segment = MagicMock()
    mock_segment.text = "Texto transcrito"
    mock_segment.start = 0.0
    mock_segment.end = 1.0
    
    mock_info = MagicMock()
    mock_info.duration = 1.0
    
    with patch('lucia.WhisperModel.return_value.transcribe') as mock_transcribe:
        mock_transcribe.return_value = ([mock_segment], mock_info)
        
        lucia.process_audio(str(audio_file), str(md_file))
    
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    
    # Check if summary is at the top and transcription follows
    assert content.startswith("---")
    assert "## Transcrição Original" in content
    assert "Texto transcrito" in content
