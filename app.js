const state = {
  rows: [],
  // Inverted index and stats for BM25
  termToDocTf: new Map(), // term -> Map(docId -> tf)
  docLengths: [], // tokens per doc
  avgDocLen: 0,
  loaded: false,
  datasetStats: {
    csv: 0,
    test: 0,
    train: 0,
    valid: 0
  },
  currentResults: [], // Store current search results for summary generation
  aiResponse: null, // Store AI-generated response
  apiUrl: 'http://localhost:8000' // Change to match your backend port
};

const formEl = document.getElementById('search-form');
const queryEl = document.getElementById('query');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const finalAnswerEl = document.getElementById('final-answer');
const summaryButtonsEl = document.getElementById('summary-buttons');
const doctorSummaryBtn = document.getElementById('doctor-summary-btn');
const patientSummaryBtn = document.getElementById('patient-summary-btn');
const summaryResultEl = document.getElementById('summary-result');

// Add new elements for AI integration
let aiResponseEl = null;
let modelStatusEl = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

async function loadAllDatasets() {
  try {
    setStatus('Loading all 4 datasets...');
    
    // Load CSV dataset
    await loadCsv();
    
    // Load JSON datasets
    await loadJsonDataset('test.json', 'test');
    await loadJsonDataset('train.json', 'train');
    await loadJsonDataset('valid.json', 'valid');
    
    buildIndex();
    state.loaded = true;
    
    const totalRows = state.rows.length;
    setStatus(`✅ Ready! Loaded ${totalRows.toLocaleString()} total entries from all 4 datasets.`);
    
    console.log('Dataset stats:', state.datasetStats);
  } catch (err) {
    console.error(err);
    setStatus('Error loading datasets. See console.');
  }
}

async function loadCsv() {
  try {
    setStatus('Loading CSV dataset...');
    const response = await fetch('./medquad.csv');
    if (!response.ok) throw new Error(`Failed to fetch medquad.csv: ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let csvText = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      csvText += decoder.decode(value, { stream: true });
      if (csvText.length > 3_000_000) {
        // Prevent memory issues in the demo; parse incrementally
        await parseChunk(csvText);
        csvText = '';
      }
    }
    if (csvText) await parseChunk(csvText, true);
    state.datasetStats.csv = state.rows.length;
    setStatus(`Loaded ${state.rows.length.toLocaleString()} CSV entries.`);
  } catch (err) {
    console.error('Error loading CSV:', err);
    setStatus('Error loading CSV dataset. See console.');
  }
}

async function loadJsonDataset(filename, datasetType) {
  try {
    setStatus(`Loading ${filename}...`);
    const response = await fetch(`./${filename}`);
    if (!response.ok) {
      console.warn(`Failed to fetch ${filename}: ${response.status}`);
      return;
    }
    
    const data = await response.json();
    console.log(`Loaded ${data.length} entries from ${filename}`);
    
    // Convert JSON entries to our format
    for (const entry of data) {
      const question = entry.question?.trim();
      const context = entry.context?.trim();
      
      if (!question) continue;
      
      // Get the best answer
      let answer = '';
      if (entry.labelled_summaries) {
        const summaries = entry.labelled_summaries;
        if (summaries.INFORMATION_SUMMARY) {
          answer = summaries.INFORMATION_SUMMARY;
        } else if (summaries.SUGGESTION_SUMMARY) {
          answer = summaries.SUGGESTION_SUMMARY;
        } else if (summaries.EXPERIENCE_SUMMARY) {
          answer = summaries.EXPERIENCE_SUMMARY;
        } else if (summaries.CAUSE_SUMMARY) {
          answer = summaries.CAUSE_SUMMARY;
        }
      }
      
      // Fallback to first answer if no summary
      if (!answer && entry.answers && entry.answers.length > 0) {
        answer = entry.answers[0];
      }
      
      if (!answer) continue;
      
      // Create row entry
      const row = {
        question: question,
        answer: answer.trim(),
        source: `Dataset: ${datasetType}`,
        focus_area: context || '',
        dataset: datasetType
      };
      
      state.rows.push(row);
    }
    
    state.datasetStats[datasetType] = data.length;
    setStatus(`Loaded ${data.length} entries from ${filename}.`);
    
  } catch (err) {
    console.error(`Error loading ${filename}:`, err);
    setStatus(`Error loading ${filename}. See console.`);
  }
}

function parseChunk(text, isLastChunk = false) {
  return new Promise((resolve) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      worker: true,
      chunk: (results) => {
        for (const row of results.data) {
          if (!row.question || !row.answer) continue;
          state.rows.push({
            question: String(row.question),
            answer: String(row.answer),
            source: row.source ? String(row.source) : 'CSV Dataset',
            focus_area: row.focus_area ? String(row.focus_area) : '',
            dataset: 'csv'
          });
        }
      },
      complete: () => resolve(),
    });
  });
}

function buildIndex() {
  state.termToDocTf = new Map();
  state.docLengths = new Array(state.rows.length).fill(0);

  for (let i = 0; i < state.rows.length; i++) {
    const row = state.rows[i];
    const tokens = [...tokenize(row.question), ...tokenize(row.answer)];
    state.docLengths[i] = tokens.length;
    const tf = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
    // populate inverted index
    for (const [t, f] of tf.entries()) {
      if (!state.termToDocTf.has(t)) state.termToDocTf.set(t, new Map());
      state.termToDocTf.get(t).set(i, f);
    }
  }
  const totalLen = state.docLengths.reduce((a, b) => a + b, 0) || 1;
  state.avgDocLen = totalLen / Math.max(1, state.rows.length);
}

function bm25Score(query, { k1 = 1.2, b = 0.75 } = {}) {
  const qTokens = tokenize(query).filter(Boolean);
  if (qTokens.length === 0) return [];
  const N = state.rows.length;
  const docScores = new Map(); // docId -> score

  for (const term of qTokens) {
    const posting = state.termToDocTf.get(term);
    if (!posting) continue;
    const df = posting.size;
    const idf = Math.log((N - df + 0.5) / (df + 0.5) + 1);
    for (const [docId, tf] of posting.entries()) {
      const dl = state.docLengths[docId] || 1;
      const denom = tf + k1 * (1 - b + b * (dl / (state.avgDocLen || 1)));
      const score = idf * ((tf * (k1 + 1)) / denom);
      docScores.set(docId, (docScores.get(docId) || 0) + score);
    }
  }

  const scores = Array.from(docScores.entries())
    .map(([i, score]) => ({ i, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((s) => ({ row: state.rows[s.i], score: s.score }));
  return scores;
}

function composeAnswer(matches) {
  if (!matches || matches.length === 0) return '';
  
  const answers = matches.map(m => m.row.answer).join(' ');
  return answers.substring(0, 500) + (answers.length > 500 ? '...' : '');
}

async function askLLM(question, contexts = []) {
  try {
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question,
        contexts: contexts,
        max_new_tokens: 200,
        temperature: 0.7
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.answer;
  } catch (error) {
    console.error('Error calling LLM:', error);
    return 'Sorry, I could not get a response from the AI model. Please try again later.';
  }
}

function displayResults(matches, query) {
  resultsEl.innerHTML = '';
  
  if (matches.length === 0) {
    resultsEl.innerHTML = '<div class="result"><p>No relevant results found.</p></div>';
    return;
  }

  for (const match of matches) {
    const div = document.createElement('div');
    div.className = 'result';
    
    // Create dataset badge
    const datasetBadge = `<span class="dataset-badge ${match.row.dataset}">${match.row.dataset.toUpperCase()}</span>`;
    
    div.innerHTML = `
      <div class="score">Relevance: ${match.score.toFixed(3)} ${datasetBadge}</div>
      <h3>${match.row.question}</h3>
      <p>${match.row.answer}</p>
      ${match.row.source ? `<small>Source: ${match.row.source}</small>` : ''}
      ${match.row.focus_area ? `<small>Context: ${match.row.focus_area}</small>` : ''}
    `;
    resultsEl.appendChild(div);
  }
  
  // Store current results for potential use
  state.currentResults = matches;
}

function displayFinalAnswer(answer) {
  finalAnswerEl.innerHTML = `
    <h3>🤖 AI Response</h3>
    <p>${answer}</p>
  `;
  finalAnswerEl.style.display = 'block';
}

// Add new function to check model status
async function checkModelStatus() {
  try {
    const response = await fetch(`${state.apiUrl}/health`);
    const data = await response.json();
    
    if (data.model_loaded) {
      setStatus(`✅ AI Model Ready: ${data.message}`);
      return true;
    } else {
      setStatus(`❌ AI Model Not Ready: ${data.message}`);
      return false;
    }
  } catch (err) {
    setStatus('❌ Cannot connect to AI model API');
    return false;
  }
}

// Add function to get AI response
async function getAIResponse(question) {
  try {
    setStatus('🤖 Generating AI response...');
    
    const response = await fetch(`${state.apiUrl}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: question,
        max_new_tokens: 300,
        temperature: 0.7
      })
    });
    
    const data = await response.json();
    console.log(`AI Response:`, data.answer);

    if (data.success) {
      state.aiResponse = data.answer;
      setStatus('✅ AI response generated successfully');
      return data; // Return the full data object
    } else {
      setStatus(`❌ AI Error: ${data.message}`);
      return null;
    }
    
  } catch (err) {
    setStatus(`❌ API Error: ${err.message}`);
    return null;
  }
}

// Add function to display AI response
function displayAIResponse(data) {
    const aiSection = document.getElementById('aiSection');
    aiSection.style.display = 'block';
    
    // Display AI response
    const aiResponseDiv = document.getElementById('aiResponse');
    aiResponseDiv.innerHTML = `
        <div class="ai-response">
            <h3>🤖 AI Medical Assistant Response</h3>
            <p>${data.answer}</p>
            <p class="model-info">Model: ${data.model_used}</p>
        </div>
    `;
}

// Add function to generate summaries using the API
async function generateSummary(summaryType) {
  if (!state.aiResponse) {
    alert('Please ask a question first to get an AI response before generating summaries.');
    return;
  }
  
  const summaryResultEl = document.getElementById('summary-result');
  const buttonEl = document.getElementById(`${summaryType}-summary-btn`);
  
  try {
    // Update button state
    buttonEl.disabled = true;
    buttonEl.innerHTML = summaryType === 'doctor' ? '👨‍⚕️ Generating...' : '👤 Generating...';
    
    setStatus(`🤖 Generating ${summaryType} summary using AI...`);
    
    // Generate summary using the API
    const response = await fetch(`${state.apiUrl}/generate-summary`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content: state.aiResponse,
        summary_type: summaryType,
        max_new_tokens: 300,
        temperature: 0.5
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.success) {
      const summaryTypeLabel = summaryType === 'doctor' ? 'Doctor Summary' : 'Patient Summary';
      const icon = summaryType === 'doctor' ? '👨‍⚕️' : '👤';
      console.log(`Generated ${summaryTypeLabel}:`, data.summary);
      summaryResultEl.innerHTML = `
        <div class="summary-output ${summaryType}-summary">
          <h5>${icon} ${summaryTypeLabel}</h5>
          <div class="summary-content">${data.summary}</div>
          <div class="model-info">
            <small>Generated by: ${data.model_used}</small>
          </div>
          <div class="summary-disclaimer">
            <small>⚠️ This is an AI-generated summary. Always consult healthcare professionals for medical decisions.</small>
          </div>
        </div>
      `;
      
      summaryResultEl.style.display = 'block';
      setStatus(`✅ ${summaryTypeLabel} generated successfully!`);
    } else {
      summaryResultEl.innerHTML = `
        <div class="summary-error">
          ❌ Error generating ${summaryType} summary: ${data.message}
        </div>
      `;
      summaryResultEl.style.display = 'block';
      setStatus(`❌ Failed to generate ${summaryType} summary`);
    }
    
  } catch (err) {
    console.error('Summary generation error:', err);
    summaryResultEl.innerHTML = `
      <div class="summary-error">
        ❌ API Error: ${err.message}
      </div>
    `;
    summaryResultEl.style.display = 'block';
    setStatus(`❌ API Error: ${err.message}`);
  } finally {
    // Reset button state
    buttonEl.disabled = false;
    buttonEl.innerHTML = summaryType === 'doctor' ? '👨‍⚕️ Doctor Summary' : '👤 Patient Summary';
  }
}

// Event listeners for summary buttons
doctorSummaryBtn.addEventListener('click', async () => {
  if (!state.aiResponse) {
    alert('Please ask a question first to get an AI response before generating summaries.');
    return;
  }
  
  await generateSummary('doctor');
});

patientSummaryBtn.addEventListener('click', async () => {
  if (!state.aiResponse) {
    alert('Please ask a question first to get an AI response before generating summaries.');
    return;
  }
  
  await generateSummary('patient');
});

// Update the search form handler
formEl.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  const query = queryEl.value.trim();
  if (!query) return;
  
  // Clear previous results
  resultsEl.innerHTML = '';
  finalAnswerEl.innerHTML = '';
  summaryButtonsEl.style.display = 'none';
  summaryResultEl.style.display = 'none';
  
  // Get AI response first
  const aiResponseData = await getAIResponse(query);
  
  // Also search through datasets for context
  if (state.loaded) {
    const results = bm25Score(query);
    displayResults(results, query);
  }
  
  // Display AI response and show summary buttons
  if (aiResponseData && aiResponseData.success) {
    displayAIResponse(aiResponseData);
    // Show summary buttons since we have an AI response
    summaryButtonsEl.style.display = 'block';
  }
});

// Initialize
loadAllDatasets();
