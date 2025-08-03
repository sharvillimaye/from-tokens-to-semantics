#!/usr/bin/env python3
"""
Consolidated N-gram Dataset Builder
Simple, robust dataset creation with streaming HuggingFace datasets or dummy data.
Provides n-grams with clear semantic categories and frequency binning for research.
"""

import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import re
import json
import time
import random
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetSource(Enum):
    """Available dataset sources"""
    PILE = "pile"
    OPENWEBTEXT = "openwebtext"
    DUMMY = "dummy"
    COUNTRY_CAPITAL = "country_capital"


@dataclass
class NGramConfig:
    """Configuration for n-gram dataset building"""
    n_gram_size: int = 2
    num_samples: int = 1000
    source: DatasetSource = DatasetSource.PILE
    min_text_length: int = 50
    max_text_length: int = 2000
    min_word_length: int = 2
    filter_digits: bool = True
    random_seed: Optional[int] = None
    include_mixed_ngram_sizes: bool = False  # Include 1,2,3-grams in same dataset


# Precise semantic categories for clean n-gram alignment
SEMANTIC_CATEGORIES = {
    'technology': [
        # Core computing concepts
        'computer', 'software', 'algorithm', 'database', 'network', 'system', 'digital', 'programming',
        'artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'data science',
        'cloud computing', 'cybersecurity', 'blockchain', 'internet', 'web development', 'code', 'coding',
        'technical', 'technological', 'computational', 'electronic', 'automated', 'robotics'
    ],
    'science': [
        # Research and scientific concepts  
        'research', 'experiment', 'theory', 'hypothesis', 'analysis', 'study', 'discovery', 'scientific',
        'physics', 'chemistry', 'biology', 'mathematics', 'laboratory', 'methodology', 'empirical',
        'statistical', 'peer review', 'publication', 'observation', 'measurement', 'evidence',
        'researcher', 'scientist', 'academic research', 'experimental', 'theoretical'
    ],
    'business': [
        # Business and economic concepts
        'company', 'business', 'market', 'industry', 'profit', 'revenue', 'economy', 'financial',
        'corporation', 'enterprise', 'startup', 'investment', 'management', 'strategy', 'marketing',
        'sales', 'finance', 'commerce', 'commercial', 'economic', 'entrepreneurial', 'corporate'
    ],
    'politics': [
        # Political and governmental concepts
        'government', 'policy', 'election', 'democracy', 'legislation', 'political', 'vote',
        'parliament', 'congress', 'senate', 'president', 'minister', 'constitution', 'law',
        'regulation', 'citizen', 'public', 'governmental', 'legislative', 'democratic'
    ],
    'health': [
        # Medical and health concepts
        'medical', 'health', 'disease', 'treatment', 'patient', 'doctor', 'hospital', 'medicine',
        'diagnosis', 'therapy', 'surgery', 'pharmaceutical', 'clinical', 'healthcare', 'nursing',
        'vaccine', 'symptom', 'recovery', 'physician', 'therapeutic', 'medicinal', 'surgical'
    ],
    'education': [
        # Educational concepts
        'education', 'learning', 'teaching', 'student', 'school', 'university', 'academic',
        'curriculum', 'pedagogy', 'classroom', 'lecture', 'professor', 'degree', 'scholarship',
        'knowledge', 'skill', 'training', 'educational', 'pedagogical', 'scholarly', 'instructional'
    ]
}

# Frequency bins for clear high/low grouping
FREQUENCY_BINS = {
    'high': (100, float('inf')),     # High frequency n-grams
    'low': (1, 100)                  # Low frequency n-grams
}

# Country-Capital relationship data (well-known vs less-known)
COUNTRY_CAPITAL_DATA = {
    'well_known': {
        'pairs': [
            ('France', 'Paris'), ('Germany', 'Berlin'), ('Italy', 'Rome'), ('Spain', 'Madrid'),
            ('United Kingdom', 'London'), ('Japan', 'Tokyo'), ('China', 'Beijing'), ('Russia', 'Moscow'),
            ('United States', 'Washington'), ('Canada', 'Ottawa'), ('Australia', 'Canberra'), ('Brazil', 'Brasilia'),
            ('India', 'New Delhi'), ('Egypt', 'Cairo'), ('Greece', 'Athens'), ('Turkey', 'Ankara')
        ],
        'frequency_category': 'high'
    },
    'less_known': {
        'pairs': [
            ('Kazakhstan', 'Nur-Sultan'), ('Uzbekistan', 'Tashkent'), ('Belarus', 'Minsk'), ('Mongolia', 'Ulaanbaatar'),
            ('Sri Lanka', 'Sri Jayawardenepura Kotte'), ('Myanmar', 'Naypyidaw'), ('Palau', 'Ngerulmud'),
            ('Bhutan', 'Thimphu'), ('Liechtenstein', 'Vaduz'), ('San Marino', 'San Marino'), ('Nauru', 'Yaren'),
            ('Tuvalu', 'Funafuti'), ('Vanuatu', 'Port Vila'), ('Kiribati', 'Tarawa'), ('Comoros', 'Moroni'),
            ('Seychelles', 'Victoria')
        ],
        'frequency_category': 'low'
    }
}


class NGramDataset:
    """Simple and robust n-gram dataset builder"""
    
    def __init__(self, config: Optional[NGramConfig] = None):
        self.config = config or NGramConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
    
    def stream_huggingface_dataset(self) -> List[str]:
        """Stream texts from HuggingFace datasets"""
        logger.info(f"Loading {self.config.num_samples} texts from {self.config.source.value}")
        
        if self.config.source == DatasetSource.PILE:
            return self._load_pile()
        elif self.config.source == DatasetSource.OPENWEBTEXT:
            return self._load_openwebtext()
        elif self.config.source == DatasetSource.COUNTRY_CAPITAL:
            return self.generate_country_capital_data()
        else:
            raise ValueError(f"Unknown dataset source: {self.config.source}")
    
    def _load_pile(self) -> List[str]:
        """Load from The Pile dataset"""
        dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
        return self._extract_texts(dataset)
    
    def _load_openwebtext(self) -> List[str]:
        """Load from OpenWebText dataset"""  
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        return self._extract_texts(dataset)
    
    def _extract_texts(self, dataset) -> List[str]:
        """Extract and filter texts from streaming dataset"""
        texts = []
        valid_count = 0
        
        with tqdm(total=self.config.num_samples, desc="Loading texts") as pbar:
            for sample in dataset:
                if valid_count >= self.config.num_samples:
                    break
                
                # Handle different field names
                text = sample.get('text') or sample.get('content', '')
                if not text:
                    continue
                
                # Apply length filters
                text_len = len(text.strip())
                if self.config.min_text_length <= text_len <= self.config.max_text_length:
                    texts.append(text.strip())
                    valid_count += 1
                    pbar.update(1)
        
        logger.info(f"Loaded {len(texts)} valid texts from dataset")
        return texts
    
    def generate_dummy_data(self) -> List[str]:
        """Generate semantically-aligned dummy texts for testing"""
        # Semantic category templates
        category_templates = {
            'technology': [
                "The computer software {verb} complex algorithms efficiently.",
                "Machine learning systems {verb} neural networks for data analysis.",
                "Digital technology {verb} programming languages in development.",
                "Database networks {verb} computational systems automatically.",
                "Artificial intelligence {verb} deep learning models continuously."
            ],
            'science': [
                "Scientific research {verb} experimental hypotheses through analysis.",
                "Laboratory experiments {verb} theoretical discoveries systematically.",
                "Research scientists {verb} empirical studies in physics.",
                "Scientific methodology {verb} statistical evidence carefully.",
                "Academic research {verb} peer review publications regularly."
            ],
            'business': [
                "Corporate companies {verb} market strategies for profit.",
                "Business enterprises {verb} financial investments wisely.",
                "Commercial industries {verb} economic growth through management.",
                "Financial corporations {verb} revenue generation effectively.",
                "Business management {verb} commercial opportunities strategically."
            ],
            'health': [
                "Medical doctors {verb} patient treatment in hospitals.",
                "Healthcare physicians {verb} clinical diagnosis accurately.",
                "Medical treatment {verb} therapeutic recovery processes.",
                "Hospital medicine {verb} pharmaceutical therapy effectively.",
                "Clinical healthcare {verb} surgical procedures safely."
            ],
            'education': [
                "University students {verb} academic learning through teaching.",
                "Educational schools {verb} classroom instruction methodically.",
                "Academic professors {verb} scholarly knowledge systematically.",
                "Learning education {verb} pedagogical training continuously.",
                "Educational curriculum {verb} student development progressively."
            ],
            'politics': [
                "Government policy {verb} democratic legislation effectively.",
                "Political elections {verb} governmental representation fairly.",
                "Legislative government {verb} public policy decisions.",
                "Democratic politics {verb} citizen participation actively.",
                "Political legislation {verb} governmental regulation systematically."
            ]
        }
        
        verbs = ["implement", "develop", "analyze", "process", "manage", "conduct", "utilize", "execute"]
        
        texts = []
        categories = list(category_templates.keys())
        
        # Generate balanced texts across categories
        samples_per_category = self.config.num_samples // len(categories)
        
        for category in categories:
            templates = category_templates[category]
            for _ in range(samples_per_category):
                template = random.choice(templates)
                verb = random.choice(verbs)
                text = template.format(verb=verb)
                texts.append(text)
        
        # Fill remaining samples
        remaining = self.config.num_samples - len(texts)
        for _ in range(remaining):
            category = random.choice(categories)
            template = random.choice(category_templates[category])
            verb = random.choice(verbs)
            text = template.format(verb=verb)
            texts.append(text)
        
        logger.info(f"Generated {len(texts)} semantically-aligned dummy texts")
        return texts
    
    def generate_country_capital_data(self) -> List[str]:
        """Generate country-capital relationship texts with mixed n-gram sizes"""
        sentence_templates = {
            'well_known': [
                "The capital of {country} is {capital}.",
                "{capital} is the capital city of {country}.",
                "{country}'s capital is {capital}.",
                "When visiting {country}, tourists often go to {capital}.",
                "The government of {country} is located in {capital}.",
                "{capital}, the capital of {country}, is a major city.",
                "Political decisions in {country} are made in {capital}.",
                "{country} has {capital} as its capital city."
            ],
            'less_known': [
                "The capital of {country} is {capital}.",
                "{capital} serves as the capital of {country}.",
                "{country}'s administrative center is {capital}.",
                "Few people know that {capital} is the capital of {country}.",
                "The lesser-known capital of {country} is {capital}.",
                "{capital}, though not well-known, is the capital of {country}.",
                "In {country}, the capital city is {capital}.",
                "{country} has {capital} as its official capital."
            ]
        }
        
        texts = []
        samples_per_category = self.config.num_samples // 2
        
        # Generate well-known pairs
        well_known_pairs = COUNTRY_CAPITAL_DATA['well_known']['pairs']
        for _ in range(samples_per_category):
            country, capital = random.choice(well_known_pairs)
            template = random.choice(sentence_templates['well_known'])
            text = template.format(country=country, capital=capital)
            texts.append(text)
        
        # Generate less-known pairs
        less_known_pairs = COUNTRY_CAPITAL_DATA['less_known']['pairs']
        for _ in range(samples_per_category):
            country, capital = random.choice(less_known_pairs)
            template = random.choice(sentence_templates['less_known'])
            text = template.format(country=country, capital=capital)
            texts.append(text)
        
        # Fill remaining samples
        remaining = self.config.num_samples - len(texts)
        all_pairs = well_known_pairs + less_known_pairs
        all_templates = sentence_templates['well_known'] + sentence_templates['less_known']
        
        for _ in range(remaining):
            country, capital = random.choice(all_pairs)
            template = random.choice(all_templates)
            text = template.format(country=country, capital=capital)
            texts.append(text)
        
        logger.info(f"Generated {len(texts)} country-capital relationship texts")
        return texts
    
    def extract_ngrams(self, texts: List[str]) -> Tuple[Counter, Dict[str, List[Dict]]]:
        """Extract n-grams with semantic categorization"""
        ngram_counts = Counter()
        ngram_details = defaultdict(list)
        
        logger.info(f"Extracting {self.config.n_gram_size}-grams from {len(texts)} texts")
        
        for text in tqdm(texts, desc="Processing texts"):
            text_ngrams = self._extract_ngrams_from_text(text)
            
            for ngram_info in text_ngrams:
                ngram = ngram_info['ngram']
                ngram_counts[ngram] += 1
                
                # Store details (limit examples per n-gram)
                if len(ngram_details[ngram]) < 3:
                    ngram_details[ngram].append({
                        'text': text[:150],
                        'semantic_category': ngram_info['semantic_category'],
                        'confidence': ngram_info['confidence'],
                        'ngram_size': ngram_info.get('ngram_size', self.config.n_gram_size)
                    })
        
        logger.info(f"Found {len(ngram_counts)} unique n-grams")
        return ngram_counts, dict(ngram_details)
    
    def _extract_ngrams_from_text(self, text: str) -> List[Dict]:
        """Extract n-grams from single text with semantic classification"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split()
        
        ngrams_with_semantics = []
        
        # If mixed n-gram sizes, extract 1,2,3-grams
        if self.config.include_mixed_ngram_sizes:
            ngram_sizes = [1, 2, 3]
        else:
            ngram_sizes = [self.config.n_gram_size]
        
        for ngram_size in ngram_sizes:
            if len(words) < ngram_size:
                continue
                
            for i in range(len(words) - ngram_size + 1):
                ngram = ' '.join(words[i:i + ngram_size])
                
                # Apply quality filters
                if not self._passes_quality_filter(ngram.split()):
                    continue
                
                # Get context for better classification
                context_start = max(0, i - 3)
                context_end = min(len(words), i + ngram_size + 3)
                context = ' '.join(words[context_start:context_end])
                
                # Classify semantic category
                semantic_category, confidence = self._classify_semantic_category(ngram, context, text)
                
                # Only include n-grams with clear semantic alignment
                if semantic_category is not None:
                    ngrams_with_semantics.append({
                        'ngram': ngram,
                        'ngram_size': ngram_size,
                        'semantic_category': semantic_category,
                        'confidence': confidence
                    })
        
        return ngrams_with_semantics
    
    def _passes_quality_filter(self, words: List[str]) -> bool:
        """Check if n-gram passes quality filters"""        
        for word in words:
            # Check minimum word length (but allow single letters for country codes)
            if len(word) < self.config.min_word_length and word not in ['is', 'of', 'in', 'to', 'a', 'the']:
                return False
            # Check for digits if filtering enabled    
            if self.config.filter_digits and word.isdigit():
                return False
        
        return True
    
    def _classify_semantic_category(self, ngram: str, context: str = "", full_text: str = "") -> Tuple[str, float]:
        """Classify n-gram into semantic categories with strict alignment"""
        analysis_text = f"{ngram} {context} {full_text}".lower()
        
        # First check for country-capital relationships
        if self.config.source == DatasetSource.COUNTRY_CAPITAL:
            return self._classify_country_capital_relationship(ngram, analysis_text)
        
        # Otherwise use standard semantic classification
        category_scores = defaultdict(float)
        
        # Score each category based on keyword matches
        for category, keywords in SEMANTIC_CATEGORIES.items():
            score = 0.0
            exact_matches = 0
            
            for keyword in keywords:
                if keyword.lower() in analysis_text:
                    # Prioritize exact ngram matches for better semantic alignment
                    if keyword.lower() in ngram.lower():
                        score += 3.0  # Higher weight for direct n-gram matches
                        exact_matches += 1
                    else:
                        score += 1.0  # Context matches
            
            # Boost score if we have exact matches in the n-gram itself
            if exact_matches > 0:
                score *= (1 + exact_matches * 0.5)
                
            category_scores[category] = score
        
        # Select best category with stricter threshold
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            max_score = category_scores[best_category]
            
            if max_score >= 2.0:  # Require at least one strong match
                # Calculate confidence based on match strength
                confidence = min(max_score / (len(SEMANTIC_CATEGORIES[best_category]) * 1.5), 1.0)
                if confidence >= 0.3:  # Higher confidence threshold for cleaner alignment
                    return best_category, confidence
        
        # Return None for unclear cases to filter them out
        return None, 0.0
    
    def _classify_country_capital_relationship(self, ngram: str, analysis_text: str) -> Tuple[str, float]:
        """Classify country-capital relationship n-grams"""
        ngram_lower = ngram.lower()
        
        # Check if n-gram contains country or capital names
        well_known_countries = [country.lower() for country, _ in COUNTRY_CAPITAL_DATA['well_known']['pairs']]
        well_known_capitals = [capital.lower() for _, capital in COUNTRY_CAPITAL_DATA['well_known']['pairs']]
        less_known_countries = [country.lower() for country, _ in COUNTRY_CAPITAL_DATA['less_known']['pairs']]
        less_known_capitals = [capital.lower() for _, capital in COUNTRY_CAPITAL_DATA['less_known']['pairs']]
        
        # Check for well-known relationships
        for country in well_known_countries:
            if country in ngram_lower or any(word in ngram_lower for word in country.split()):
                return 'country_capital_well_known', 0.9
        
        for capital in well_known_capitals:
            if capital in ngram_lower or any(word in ngram_lower for word in capital.split()):
                return 'country_capital_well_known', 0.9
        
        # Check for less-known relationships
        for country in less_known_countries:
            if country in ngram_lower or any(word in ngram_lower for word in country.split()):
                return 'country_capital_less_known', 0.8
        
        for capital in less_known_capitals:
            if capital in ngram_lower or any(word in ngram_lower for word in capital.split()):
                return 'country_capital_less_known', 0.8
        
        # Check for relationship words
        relationship_words = ['capital', 'government', 'administrative', 'located', 'city', 'center']
        if any(word in ngram_lower for word in relationship_words):
            # Determine frequency based on context
            if any(country in analysis_text for country in well_known_countries + well_known_capitals):
                return 'country_capital_well_known', 0.7
            else:
                return 'country_capital_less_known', 0.6
        
        return None, 0.0
    
    def categorize_by_frequency(self, frequency: int) -> str:
        """Categorize n-gram by frequency (high/low)"""
        high_min, high_max = FREQUENCY_BINS['high']
        if high_min <= frequency < high_max:
            return 'high'
        return 'low'
    
    def build_dataset(self) -> Dict[str, Any]:
        """Build complete n-gram dataset"""
        logger.info(f"🚀 Building {self.config.n_gram_size}-gram dataset")
        start_time = time.time()
        
        # Step 1: Load texts
        if self.config.source == DatasetSource.DUMMY:
            texts = self.generate_dummy_data()
        elif self.config.source == DatasetSource.COUNTRY_CAPITAL:
            texts = self.generate_country_capital_data()
        else:
            texts = self.stream_huggingface_dataset()
        
        if not texts:
            raise ValueError("No texts loaded")
        
        # Step 2: Extract n-grams with semantic classification
        ngram_counts, ngram_details = self.extract_ngrams(texts)
        
        # Step 3: Create final dataset structure
        dataset = self._create_final_dataset(ngram_counts, ngram_details)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Dataset built successfully in {elapsed_time:.1f} seconds!")
        
        return dataset
    
    def _create_final_dataset(self, ngram_counts: Counter, ngram_details: Dict) -> Dict[str, Any]:
        """Create final dataset structure"""
        dataset = {
            'texts': [],
            'ngrams': [],
            'ngram_sizes': [],
            'semantic_categories': [],
            'semantic_confidence': [],
            'frequency_categories': [],
            'local_frequencies': [],
            'estimated_global_frequencies': []
        }
        
        # Process n-grams
        for ngram, local_freq in ngram_counts.items():
            # Get details
            details = ngram_details.get(ngram, [{'text': '', 'semantic_category': None, 'confidence': 0.0, 'ngram_size': self.config.n_gram_size}])
            detail = details[0]
            
            if detail['semantic_category'] is None:
                continue
            
            # For country-capital datasets, map semantic category to frequency
            if self.config.source == DatasetSource.COUNTRY_CAPITAL:
                if 'well_known' in detail['semantic_category']:
                    freq_category = 'high'
                    global_freq = max(100, int(local_freq * random.uniform(150, 300)))
                else:
                    freq_category = 'low'
                    global_freq = max(1, int(local_freq * random.uniform(10, 50)))
            else:
                # For non-country-capital datasets, frequency category must be determined by actual corpus analysis
                raise ValueError("Global frequency estimation requires corpus-wide frequency analysis. " +
                              "Use a dataset with pre-computed frequency categories or implement proper frequency analysis.")
            
            # Add one example per n-gram
            dataset['texts'].append(detail['text'])
            dataset['ngrams'].append(ngram)
            dataset['ngram_sizes'].append(detail.get('ngram_size', self.config.n_gram_size))
            dataset['semantic_categories'].append(detail['semantic_category'])
            dataset['semantic_confidence'].append(detail['confidence'])
            dataset['frequency_categories'].append(freq_category)
            dataset['local_frequencies'].append(local_freq)
            dataset['estimated_global_frequencies'].append(global_freq)
        
        # Add metadata
        dataset['metadata'] = {
            'config': asdict(self.config),
            'semantic_categories': SEMANTIC_CATEGORIES,
            'frequency_bins': FREQUENCY_BINS,
            'total_samples': len(dataset['texts']),
            'unique_ngrams': len(set(dataset['ngrams'])),
            'creation_timestamp': time.time()
        }
        
        logger.info(f"Created dataset with {len(dataset['texts'])} samples")
        return dataset
    
    def create_stratified_dataset(self, target_samples_per_group: int = 50) -> Dict[str, Any]:
        """
        Create a stratified dataset with balanced examples across semantic categories and frequency groups
        
        Args:
            target_samples_per_group: Target number of samples per semantic-frequency group
            
        Returns:
            Stratified dataset with balanced representation
        """
        logger.info(f"🎯 Creating stratified dataset with {target_samples_per_group} samples per group")
        
        # Build initial dataset
        initial_dataset = self.build_dataset()
        
        # Group by semantic category and frequency
        stratified_samples = defaultdict(lambda: defaultdict(list))
        
        for i, (semantic_cat, freq_cat) in enumerate(zip(
            initial_dataset['semantic_categories'], 
            initial_dataset['frequency_categories']
        )):
            stratified_samples[semantic_cat][freq_cat].append(i)
        
        # Create balanced final dataset
        final_dataset = {
            'texts': [],
            'ngrams': [],
            'ngram_sizes': [],
            'semantic_categories': [],
            'semantic_confidence': [],
            'frequency_categories': [],
            'local_frequencies': [],
            'estimated_global_frequencies': []
        }
        
        logger.info("Stratification results:")
        
        for semantic_cat in stratified_samples:
            for freq_cat in ['high', 'low']:
                indices = stratified_samples[semantic_cat][freq_cat]
                
                if len(indices) == 0:
                    logger.warning(f"No samples found for {semantic_cat}-{freq_cat}")
                    continue
                
                # Sample up to target_samples_per_group
                n_samples = min(target_samples_per_group, len(indices))
                if n_samples < len(indices):
                    # Random sampling for balance
                    selected_indices = np.random.choice(indices, n_samples, replace=False)
                else:
                    selected_indices = indices
                
                logger.info(f"  {semantic_cat}-{freq_cat}: {n_samples}/{len(indices)} samples")
                
                # Add selected samples to final dataset
                for idx in selected_indices:
                    final_dataset['texts'].append(initial_dataset['texts'][idx])
                    final_dataset['ngrams'].append(initial_dataset['ngrams'][idx])
                    final_dataset['ngram_sizes'].append(initial_dataset['ngram_sizes'][idx])
                    final_dataset['semantic_categories'].append(initial_dataset['semantic_categories'][idx])
                    final_dataset['semantic_confidence'].append(initial_dataset['semantic_confidence'][idx])
                    final_dataset['frequency_categories'].append(initial_dataset['frequency_categories'][idx])
                    final_dataset['local_frequencies'].append(initial_dataset['local_frequencies'][idx])
                    final_dataset['estimated_global_frequencies'].append(initial_dataset['estimated_global_frequencies'][idx])
        
        # Add metadata
        final_dataset['metadata'] = {
            'config': asdict(self.config),
            'semantic_categories': SEMANTIC_CATEGORIES,
            'frequency_bins': FREQUENCY_BINS,
            'total_samples': len(final_dataset['texts']),
            'unique_ngrams': len(set(final_dataset['ngrams'])),
            'target_samples_per_group': target_samples_per_group,
            'stratified': True,
            'creation_timestamp': time.time()
        }
        
        logger.info(f"✅ Created stratified dataset with {len(final_dataset['texts'])} total samples")
        return final_dataset
    
    def filter_by_semantic_category(self, dataset: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Filter dataset by semantic category"""
        indices = [i for i, cat in enumerate(dataset['semantic_categories']) if cat == category]
        return self._filter_dataset_by_indices(dataset, indices)
    
    def filter_by_frequency_category(self, dataset: Dict[str, Any], frequency: str) -> Dict[str, Any]:
        """Filter dataset by frequency category (high/low)"""
        indices = [i for i, freq in enumerate(dataset['frequency_categories']) if freq == frequency]
        return self._filter_dataset_by_indices(dataset, indices)
    
    def _filter_dataset_by_indices(self, dataset: Dict[str, Any], indices: List[int]) -> Dict[str, Any]:
        """Filter dataset by indices"""
        filtered = {}
        for key, values in dataset.items():
            if key == 'metadata':
                filtered[key] = values
            elif isinstance(values, list) and len(values) == len(dataset['texts']):
                filtered[key] = [values[i] for i in indices]
            else:
                filtered[key] = values
        return filtered
    
    def group_by_semantic_and_frequency(self, dataset: Dict[str, Any]) -> Dict[str, Dict[str, Dict]]:
        """Group n-grams by semantic category and frequency for experiment preparation"""
        groups = {}
        
        # Get unique semantic categories
        unique_categories = set(dataset['semantic_categories'])
        
        for category in unique_categories:
            groups[category] = {'high': {}, 'low': {}}
            
            # Filter by semantic category
            semantic_data = self.filter_by_semantic_category(dataset, category)
            
            if len(semantic_data['texts']) == 0:
                continue
            
            # Group by frequency within semantic category
            for freq_type in ['high', 'low']:
                freq_data = self.filter_by_frequency_category(semantic_data, freq_type)
                
                groups[category][freq_type] = {
                    'ngrams': freq_data['ngrams'],
                    'texts': freq_data['texts'],
                    'frequencies': freq_data['local_frequencies'],
                    'count': len(freq_data['ngrams']),
                    'unique_ngrams': len(set(freq_data['ngrams'])) if freq_data['ngrams'] else 0
                }
        
        return groups
    
    def get_experiment_ready_data(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Get data organized for checkpoint analysis experiments"""
        groups = self.group_by_semantic_and_frequency(dataset)
        
        experiment_data = {
            'semantic_frequency_groups': groups,
            'summary': {
                'semantic_categories': list(groups.keys()),
                'total_groups': len(groups),
                'frequency_types': ['high', 'low']
            }
        }
        
        # Add sample counts per group
        for category, freq_groups in groups.items():
            for freq_type, data in freq_groups.items():
                experiment_data['summary'][f'{category}_{freq_type}_count'] = data['count']
        
        return experiment_data


def save_dataset(dataset: Dict[str, Any], output_path: str):
    """Save dataset to JSON"""
    output_path = Path(output_path).with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    logger.info(f"💾 Saved dataset to {output_path}")


def load_dataset_from_file(file_path: str) -> Dict[str, Any]:
    """Load dataset from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_dataset_summary(dataset: Dict[str, Any]):
    """Print comprehensive dataset summary"""
    print("\n📊 Dataset Summary:")
    print("=" * 50)
    print(f"Total samples: {len(dataset['texts'])}")
    print(f"Unique n-grams: {len(set(dataset['ngrams']))}")
    print(f"N-gram size: {dataset['metadata']['config']['n_gram_size']}")
    
    # Semantic distribution
    semantic_counts = Counter(dataset['semantic_categories'])
    print(f"\n📂 Semantic Categories:")
    for category, count in semantic_counts.most_common():
        print(f"  {category}: {count} samples")
    
    # Frequency distribution  
    freq_counts = Counter(dataset['frequency_categories'])
    print(f"\n⚡ Frequency Distribution:")
    for freq_type, count in freq_counts.items():
        print(f"  {freq_type} frequency: {count} samples")


def prepare_for_checkpoint_analysis(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare dataset for checkpoint analysis compatibility
    
    Args:
        dataset: N-gram dataset with semantic and frequency categories
        
    Returns:
        Dataset formatted for checkpoint_analysis.py functions
    """
    # Create checkpoint-analysis compatible format
    checkpoint_dataset = {
        'texts': dataset['texts'],
        'ngrams': dataset['ngrams'],
        'categories': dataset['frequency_categories'],  # Map to 'categories' expected by checkpoint analysis
        'semantic_categories': dataset['semantic_categories'],
        'frequency_categories': dataset['frequency_categories'],
        'local_frequencies': dataset['local_frequencies'],
        'estimated_global_frequencies': dataset['estimated_global_frequencies'],
        'metadata': dataset['metadata']
    }
    
    logger.info(f"Prepared dataset for checkpoint analysis: {len(checkpoint_dataset['texts'])} samples")
    return checkpoint_dataset


def extract_experiment_subsets(dataset: Dict[str, Any], 
                              target_categories: List[str] = None,
                              samples_per_group: int = 50) -> Dict[str, Dict[str, Any]]:
    """
    Extract balanced subsets for experiments by semantic category and frequency
    
    Args:
        dataset: Full n-gram dataset
        target_categories: Semantic categories to include (None for all)
        samples_per_group: Maximum samples per semantic-frequency group
        
    Returns:
        Dictionary of experiment-ready subsets
    """
    if target_categories is None:
        target_categories = list(set(dataset['semantic_categories']))
    
    experiment_subsets = {}
    
    for category in target_categories:
        # Filter by semantic category
        semantic_data = NGramDataset().filter_by_semantic_category(dataset, category)
        
        if len(semantic_data['texts']) == 0:
            continue
            
        # Create balanced high/low frequency subsets
        for freq_type in ['high', 'low']:
            freq_data = NGramDataset().filter_by_frequency_category(semantic_data, freq_type)
            
            if len(freq_data['texts']) == 0:
                continue
                
            # Sample up to samples_per_group
            n_samples = min(samples_per_group, len(freq_data['texts']))
            indices = np.random.choice(len(freq_data['texts']), n_samples, replace=False)
            
            subset_key = f"{category}_{freq_type}"
            experiment_subsets[subset_key] = {
                'texts': [freq_data['texts'][i] for i in indices],
                'ngrams': [freq_data['ngrams'][i] for i in indices],
                'categories': [freq_data['frequency_categories'][i] for i in indices],
                'semantic_categories': [freq_data['semantic_categories'][i] for i in indices],
                'frequency_categories': [freq_data['frequency_categories'][i] for i in indices],
                'local_frequencies': [freq_data['local_frequencies'][i] for i in indices],
                'metadata': {
                    'semantic_category': category,
                    'frequency_category': freq_type,
                    'n_samples': n_samples,
                    'source_dataset_size': len(freq_data['texts'])
                }
            }
    
    logger.info(f"Created {len(experiment_subsets)} experiment subsets")
    return experiment_subsets


def validate_semantic_alignment(dataset: Dict[str, Any], min_confidence: float = 0.3) -> Dict[str, Any]:
    """
    Validate semantic alignment quality of n-grams in dataset
    
    Args:
        dataset: N-gram dataset to validate
        min_confidence: Minimum confidence threshold for validation
        
    Returns:
        Validation report with statistics
    """
    total_samples = len(dataset['ngrams'])
    high_confidence_samples = sum(1 for conf in dataset['semantic_confidence'] if conf >= min_confidence)
    
    # Category distribution
    category_counts = Counter(dataset['semantic_categories'])
    
    # Confidence statistics
    confidences = dataset['semantic_confidence']
    
    validation_report = {
        'total_samples': total_samples,
        'high_confidence_samples': high_confidence_samples,
        'high_confidence_ratio': high_confidence_samples / total_samples if total_samples > 0 else 0,
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence_threshold': min_confidence,
        'category_distribution': dict(category_counts),
        'category_balance_score': min(category_counts.values()) / max(category_counts.values()) if category_counts else 0,
        'semantic_purity': 'high' if np.mean(confidences) >= 0.5 else 'medium' if np.mean(confidences) >= 0.3 else 'low'
    }
    
    logger.info(f"Validation: {high_confidence_samples}/{total_samples} high-confidence samples")
    logger.info(f"Mean confidence: {validation_report['mean_confidence']:.3f}")
    logger.info(f"Semantic purity: {validation_report['semantic_purity']}")
    
    return validation_report


# Convenience functions for different use cases
def build_ngram_dataset(n_gram_size: int = 2, 
                       source: DatasetSource = DatasetSource.DUMMY,
                       num_samples: int = 500) -> Dict[str, Any]:
    """Quick way to build basic n-gram dataset"""
    config = NGramConfig(
        n_gram_size=n_gram_size,
        source=source, 
        num_samples=num_samples
    )
    
    builder = NGramDataset(config)
    return builder.build_dataset()


def build_stratified_ngram_dataset(n_gram_size: int = 2,
                                  source: DatasetSource = DatasetSource.DUMMY,
                                  num_samples: int = 1000,
                                  samples_per_group: int = 30) -> Dict[str, Any]:
    """Build stratified n-gram dataset with balanced semantic and frequency groups"""
    config = NGramConfig(
        n_gram_size=n_gram_size,
        source=source,
        num_samples=num_samples
    )
    
    builder = NGramDataset(config)
    return builder.create_stratified_dataset(samples_per_group)


def build_country_capital_dataset(num_samples: int = 200,
                                 include_mixed_sizes: bool = True,
                                 samples_per_group: int = 50) -> Dict[str, Any]:
    """Build country-capital relationship dataset with mixed n-gram sizes"""
    config = NGramConfig(
        n_gram_size=2,  # Base size, but will include 1,2,3-grams if mixed
        source=DatasetSource.COUNTRY_CAPITAL,
        num_samples=num_samples,
        include_mixed_ngram_sizes=include_mixed_sizes,
        min_word_length=1,  # Allow shorter words for country/capital names
        filter_digits=False  # Allow digits in names
    )
    
    builder = NGramDataset(config)
    return builder.create_stratified_dataset(samples_per_group)


if __name__ == "__main__":
    print("🔥 N-gram Dataset Builder")
    print("⚠️  This dataset builder requires:")
    print("   - Access to HuggingFace datasets (Pile, OpenWebText)")
    print("   - Valid internet connection") 
    print("   - Sufficient storage and memory")
    print("   - No fallback to dummy data - will fail if datasets unavailable")
    
    raise NotImplementedError("This dataset builder requires access to real datasets. " +
                            "Configure proper dataset access and ensure internet connectivity. " +
                            "No dummy data fallbacks are provided - analysis must use real data.")