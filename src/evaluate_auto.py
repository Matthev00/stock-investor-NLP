import json
import os
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List
import time

class AutoReportEvaluator:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.evaluation_results = []
        
    def calculate_readability_score(self, text: str) -> float:
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        syllables = sum(self._count_syllables(word) for word in text.split())
        
        if sentences == 0 or words == 0:
            return 0
        
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        if syllable_count == 0:
            syllable_count = 1
            
        return syllable_count
    
    def analyze_sentiment_distribution(self, text: str) -> Dict[str, int]:  
        positive_words = [
            'growth', 'strong', 'increase', 'profit', 'gain', 'bullish', 
            'opportunity', 'positive', 'success', 'excellent', 'outperform',
            'buy', 'upside', 'momentum', 'robust', 'solid'
        ]
        
        negative_words = [
            'decline', 'weak', 'decrease', 'loss', 'risk', 'bearish',
            'concern', 'negative', 'fail', 'poor', 'underperform',
            'sell', 'downside', 'volatility', 'threat', 'challenge'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        total = positive_count + negative_count
        sentiment_balance = (positive_count - negative_count) / total if total > 0 else 0
        
        return {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'sentiment_balance': sentiment_balance,
            'sentiment_label': 'Bullish' if sentiment_balance > 0.2 else 'Bearish' if sentiment_balance < -0.2 else 'Neutral'
        }
    
    def analyze_data_richness(self, text: str) -> Dict[str, any]:
        numbers = re.findall(r'\b\d+\.?\d*%?\b', text)
        percentages = re.findall(r'\d+\.?\d*%', text)
        dollar_amounts = re.findall(r'\$\d+\.?\d*[BMK]?', text)
        
        dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text)
        
        financial_metrics = [
            'P/E', 'P/S', 'EPS', 'ROE', 'ROA', 'EBITDA', 'Revenue', 
            'Market Cap', 'Dividend', 'Beta', 'RSI', 'MACD', 'SMA', 'EMA'
        ]
        metrics_found = [metric for metric in financial_metrics if metric in text]
        
        citations = len(re.findall(r'according to|source:|based on|reported by', text, re.IGNORECASE))
        
        return {
            'total_numbers': len(numbers),
            'percentages': len(percentages),
            'dollar_amounts': len(dollar_amounts),
            'dates_mentioned': len(dates),
            'financial_metrics_count': len(metrics_found),
            'financial_metrics': metrics_found,
            'citations': citations,
            'data_density': (len(numbers) + len(dates) + citations) / len(text.split()) if len(text.split()) > 0 else 0
        }
    
    def analyze_structure_quality(self, text: str) -> Dict[str, any]:
        required_sections = {
            'Executive Summary': r'executive summary|overview',
            'Sentiment Analysis': r'sentiment|market sentiment|public opinion',
            'Technical Analysis': r'technical analysis|technical indicators|chart analysis',
            'Fundamental Analysis': r'fundamental analysis|financial analysis|company analysis',
            'Risk Factors': r'risk|risks|risk factors|concerns',
            'Investment Outlook': r'outlook|forecast|projection|future',
            'Recommendation': r'recommendation|rating|action|verdict'
        }
        
        sections_found = {}
        for section_name, pattern in required_sections.items():
            if re.search(pattern, text, re.IGNORECASE):
                sections_found[section_name] = True
            else:
                sections_found[section_name] = False
        
        headers = len(re.findall(r'^#{1,3}\s+.+$', text, re.MULTILINE))
        bullet_points = text.count('•') + text.count('*') + text.count('-')
        
        paragraphs = len([p for p in text.split('\n\n') if len(p.strip()) > 50])
        
        sections_coverage = sum(sections_found.values()) / len(sections_found)
        
        return {
            'sections_found': sections_found,
            'sections_coverage_percent': sections_coverage * 100,
            'total_sections': sum(sections_found.values()),
            'headers_count': headers,
            'bullet_points': bullet_points,
            'paragraphs': paragraphs,
            'structure_score': (sections_coverage * 0.6 + min(headers/10, 1) * 0.2 + min(paragraphs/15, 1) * 0.2) * 100
        }
    
    def analyze_actionability(self, text: str) -> Dict[str, any]:
        action_words = {
            'buy': ['buy', 'purchase', 'accumulate', 'long position'],
            'sell': ['sell', 'short', 'exit', 'reduce position'],
            'hold': ['hold', 'maintain', 'keep', 'neutral']
        }
        
        text_lower = text.lower()
        actions_found = {}
        
        for action, keywords in action_words.items():
            count = sum(text_lower.count(keyword) for keyword in keywords)
            actions_found[action] = count

        price_targets = re.findall(r'price target|target price|fair value', text, re.IGNORECASE)
        specific_prices = re.findall(r'\$\d+\.?\d*', text)
        
        time_horizons = re.findall(r'\d+[-\s](?:day|week|month|year)|short[- ]term|medium[- ]term|long[- ]term', text, re.IGNORECASE)

        has_clear_recommendation = any(actions_found.values()) and (len(price_targets) > 0 or len(time_horizons) > 0)
        
        return {
            'buy_mentions': actions_found['buy'],
            'sell_mentions': actions_found['sell'],
            'hold_mentions': actions_found['hold'],
            'primary_action': max(actions_found, key=actions_found.get) if any(actions_found.values()) else 'None',
            'has_price_target': len(price_targets) > 0,
            'price_targets_count': len(price_targets),
            'has_time_horizon': len(time_horizons) > 0,
            'time_horizons_mentioned': len(time_horizons),
            'has_clear_recommendation': has_clear_recommendation,
            'actionability_score': (
                (1 if has_clear_recommendation else 0) * 40 +
                (min(len(price_targets), 2) / 2) * 30 +
                (min(len(time_horizons), 2) / 2) * 30
            )
        }
    
    def calculate_overall_quality_score(self, metrics: Dict) -> float:
        weights = {
            'structure': 0.25,
            'data_richness': 0.25,
            'readability': 0.20,
            'actionability': 0.20,
            'balance': 0.10
        }

        structure_score = metrics['structure']['structure_score']
        
        data_score = min(100, (
            metrics['data_richness']['total_numbers'] * 0.5 +
            metrics['data_richness']['financial_metrics_count'] * 5 +
            metrics['data_richness']['citations'] * 10
        ))
        
        readability_score = metrics['readability']['readability_score']
        
        actionability_score = metrics['actionability']['actionability_score']

        sentiment_balance = abs(metrics['sentiment']['sentiment_balance'])
        balance_score = (1 - min(sentiment_balance, 1)) * 100
        
        overall_score = (
            structure_score * weights['structure'] +
            data_score * weights['data_richness'] +
            readability_score * weights['readability'] +
            actionability_score * weights['actionability'] +
            balance_score * weights['balance']
        )
        
        return round(overall_score, 2)
    
    def evaluate_report(self, stock_symbol: str) -> Dict:
        report_path = self.results_dir / f"{stock_symbol}.md"
        
        if not report_path.exists():
            print(f"Nie znaleziono raportu: {report_path}")
            return None
        
        print(f"\n{'='*80}")
        print(f"EWALUACJA: {stock_symbol}")
        print(f"{'='*80}\n")
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report_text = f.read()
        
        start_time = time.time()

        word_count = len(report_text.split())
        char_count = len(report_text)
        line_count = len(report_text.split('\n'))
        
        print(f"Podstawowe statystyki:")
        print(f"   • Słowa: {word_count}")
        print(f"   • Znaki: {char_count}")
        print(f"   • Linie: {line_count}\n")
        
        print("Analiza struktury...")
        structure_metrics = self.analyze_structure_quality(report_text)
        
        print("Analiza bogactwa danych...")
        data_metrics = self.analyze_data_richness(report_text)
        
        print("Analiza czytelności...")
        readability_score = self.calculate_readability_score(report_text)
        
        print("Analiza sentymentu...")
        sentiment_metrics = self.analyze_sentiment_distribution(report_text)
        
        print("Analiza użyteczności...")
        actionability_metrics = self.analyze_actionability(report_text)
        
        all_metrics = {
            'basic': {
                'word_count': word_count,
                'char_count': char_count,
                'line_count': line_count
            },
            'structure': structure_metrics,
            'data_richness': data_metrics,
            'readability': {
                'readability_score': readability_score,
                'difficulty_level': self._get_readability_level(readability_score)
            },
            'sentiment': sentiment_metrics,
            'actionability': actionability_metrics
        }

        overall_score = self.calculate_overall_quality_score(all_metrics)
        
        evaluation_time = time.time() - start_time
        
        result = {
            'stock_symbol': stock_symbol,
            'evaluated_at': datetime.now().isoformat(),
            'evaluation_time_seconds': round(evaluation_time, 2),
            'metrics': all_metrics,
            'overall_quality_score': overall_score,
            'quality_grade': self._get_quality_grade(overall_score),
            'report_file': str(report_path)
        }
        
        self.evaluation_results.append(result)

        self._print_summary(result)
        
        return result
    
    def _get_readability_level(self, score: float) -> str:
        if score >= 80:
            return "Very Easy"
        elif score >= 60:
            return "Easy"
        elif score >= 50:
            return "Fairly Easy"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def _get_quality_grade(self, score: float) -> str:
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Satisfactory)"
        elif score >= 60:
            return "D (Needs Improvement)"
        else:
            return "F (Poor)"
    
    def _print_summary(self, result: Dict):

        metrics = result['metrics']
        
        print(f"\n{'─'*80}")
        print(f"WYNIKI EWALUACJI")
        print(f"{'─'*80}\n")
        
        print(f"OGÓLNA OCENA JAKOŚCI: {result['overall_quality_score']}/100")
        print(f"   Ocena: {result['quality_grade']}\n")

        print(f"STRUKTURA ({metrics['structure']['structure_score']:.1f}/100):")
        print(f"   • Pokrycie sekcji: {metrics['structure']['sections_coverage_percent']:.1f}%")
        print(f"   • Znalezione sekcje: {metrics['structure']['total_sections']}/7")
        print(f"   • Nagłówki: {metrics['structure']['headers_count']}")
        print(f"   • Paragrafy: {metrics['structure']['paragraphs']}\n")

        print(f"BOGACTWO DANYCH:")
        print(f"   • Liczby: {metrics['data_richness']['total_numbers']}")
        print(f"   • Procentowe wartości: {metrics['data_richness']['percentages']}")
        print(f"   • Kwoty w $: {metrics['data_richness']['dollar_amounts']}")
        print(f"   • Wskaźniki finansowe: {metrics['data_richness']['financial_metrics_count']}")
        print(f"   • Cytowania źródeł: {metrics['data_richness']['citations']}")
        print(f"   • Gęstość danych: {metrics['data_richness']['data_density']:.4f}\n")
        
        print(f"CZYTELNOŚĆ:")
        print(f"   • Wynik: {metrics['readability']['readability_score']:.1f}/100")
        print(f"   • Poziom: {metrics['readability']['difficulty_level']}\n")
        
        print(f"ANALIZA SENTYMENTU:")
        print(f"   • Słowa pozytywne: {metrics['sentiment']['positive_words']}")
        print(f"   • Słowa negatywne: {metrics['sentiment']['negative_words']}")
        print(f"   • Balans: {metrics['sentiment']['sentiment_balance']:.2f}")
        print(f"   • Wydźwięk: {metrics['sentiment']['sentiment_label']}\n")

        print(f"UŻYTECZNOŚĆ ({metrics['actionability']['actionability_score']:.1f}/100):")
        print(f"   • Główna rekomendacja: {metrics['actionability']['primary_action']}")
        print(f"   • Wzmianka 'Buy': {metrics['actionability']['buy_mentions']}x")
        print(f"   • Wzmianka 'Hold': {metrics['actionability']['hold_mentions']}x")
        print(f"   • Wzmianka 'Sell': {metrics['actionability']['sell_mentions']}x")
        print(f"   • Cena docelowa: {'✓' if metrics['actionability']['has_price_target'] else '✗'}")
        print(f"   • Horyzont czasowy: {'✓' if metrics['actionability']['has_time_horizon'] else '✗'}")
        print(f"   • Jasna rekomendacja: {'✓' if metrics['actionability']['has_clear_recommendation'] else '✗'}\n")
        
        print(f"Czas ewaluacji: {result['evaluation_time_seconds']}s")
        print(f"{'─'*80}\n")
    
    def save_results(self, output_file="evaluation_results_auto.json"):
        output_path = self.results_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
        
        print(f"Wyniki zapisane: {output_path}")
    
    def generate_comparison_table(self):
        if not self.evaluation_results:
            print("Brak wyników do porównania")
            return
        
        output_path = self.results_dir / "evaluation_comparison_auto.md"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Automatyczne Porównanie Ewaluacji Raportów\n\n")
            f.write(f"*Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            avg_score = sum(r['overall_quality_score'] for r in self.evaluation_results) / len(self.evaluation_results)
            f.write(f"## Podsumowanie\n\n")
            f.write(f"- **Liczba raportów:** {len(self.evaluation_results)}\n")
            f.write(f"- **Średnia ocena jakości:** {avg_score:.2f}/100\n")
            f.write(f"- **Najwyżej oceniony:** {max(self.evaluation_results, key=lambda x: x['overall_quality_score'])['stock_symbol']} ")
            f.write(f"({max(r['overall_quality_score'] for r in self.evaluation_results):.2f})\n")
            f.write(f"- **Najniżej oceniony:** {min(self.evaluation_results, key=lambda x: x['overall_quality_score'])['stock_symbol']} ")
            f.write(f"({min(r['overall_quality_score'] for r in self.evaluation_results):.2f})\n\n")
            
            f.write("## Porównanie Ogólne\n\n")
            f.write("| Symbol | Ocena Ogólna | Grade | Słowa | Struktura | Użyteczność | Sentyment |\n")
            f.write("|--------|--------------|-------|-------|-----------|-------------|----------|\n")
            
            for result in sorted(self.evaluation_results, key=lambda x: x['overall_quality_score'], reverse=True):
                symbol = result['stock_symbol']
                score = result['overall_quality_score']
                grade = result['quality_grade']
                words = result['metrics']['basic']['word_count']
                structure = result['metrics']['structure']['structure_score']
                actionability = result['metrics']['actionability']['actionability_score']
                sentiment = result['metrics']['sentiment']['sentiment_label']
                
                f.write(f"| {symbol} | {score:.1f} | {grade} | {words} | "
                       f"{structure:.1f} | {actionability:.1f} | {sentiment} |\n")
            
            f.write("\n## Szczegółowe Metryki\n\n")
            f.write("| Symbol | Sekcje | Dane Liczbowe | Wskaźniki | Cytowania | Czytelność |\n")
            f.write("|--------|--------|---------------|-----------|-----------|------------|\n")
            
            for result in self.evaluation_results:
                symbol = result['stock_symbol']
                sections = result['metrics']['structure']['sections_coverage_percent']
                numbers = result['metrics']['data_richness']['total_numbers']
                metrics_count = result['metrics']['data_richness']['financial_metrics_count']
                citations = result['metrics']['data_richness']['citations']
                readability = result['metrics']['readability']['readability_score']
                
                f.write(f"| {symbol} | {sections:.0f}% | {numbers} | {metrics_count} | "
                       f"{citations} | {readability:.1f} |\n")
            
            f.write("\n## Rekomendacje\n\n")
            f.write("| Symbol | Główna Akcja | Cena Docelowa | Horyzont | Jasność |\n")
            f.write("|--------|--------------|---------------|----------|--------|\n")
            
            for result in self.evaluation_results:
                symbol = result['stock_symbol']
                action = result['metrics']['actionability']['primary_action']
                price_target = '✓' if result['metrics']['actionability']['has_price_target'] else '✗'
                time_horizon = '✓' if result['metrics']['actionability']['has_time_horizon'] else '✗'
                clear = '✓' if result['metrics']['actionability']['has_clear_recommendation'] else '✗'
                
                f.write(f"| {symbol} | {action} | {price_target} | {time_horizon} | {clear} |\n")
        
        print(f"Tabela porównawcza: {output_path}")
    
    def generate_visualization_data(self):
        """Generuje dane do wizualizacji (opcjonalne)"""
        
        if not self.evaluation_results:
            return
        
        output_path = self.results_dir / "evaluation_viz_data.json"
        
        viz_data = {
            'labels': [r['stock_symbol'] for r in self.evaluation_results],
            'overall_scores': [r['overall_quality_score'] for r in self.evaluation_results],
            'structure_scores': [r['metrics']['structure']['structure_score'] for r in self.evaluation_results],
            'actionability_scores': [r['metrics']['actionability']['actionability_score'] for r in self.evaluation_results],
            'readability_scores': [r['metrics']['readability']['readability_score'] for r in self.evaluation_results],
            'word_counts': [r['metrics']['basic']['word_count'] for r in self.evaluation_results],
            'data_density': [r['metrics']['data_richness']['data_density'] for r in self.evaluation_results]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2)
        
        print(f"Dane do wizualizacji: {output_path}")


def main():
    """Główna funkcja programu"""
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║   SYSTEM AUTOMATYCZNEJ EWALUACJI RAPORTÓW INWESTYCYJNYCH    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    evaluator = AutoReportEvaluator()
    
    available_reports = list(evaluator.results_dir.glob("*.md"))
    
    if not available_reports:
        print(f"Nie znaleziono raportów w {evaluator.results_dir}/")
        print("   Najpierw wygeneruj raporty używając: streamlit run src/app.py")
        return
    
    print(f"Znaleziono {len(available_reports)} raportów:\n")
    for i, report in enumerate(available_reports, 1):
        print(f"   {i}. {report.stem}")
    
    print("\n" + "="*80)
    print("Rozpoczynam automatyczną ewaluację wszystkich raportów...")
    print("="*80)
    
    start_time = time.time()

    for report in available_reports:
        try:
            evaluator.evaluate_report(report.stem)
        except Exception as e:
            print(f"\n Błąd podczas ewaluacji {report.stem}: {e}\n")
            continue
    
    total_time = time.time() - start_time
    
    if evaluator.evaluation_results:
        print("\n" + "="*80)
        print("Zapisywanie wyników...")
        print("="*80 + "\n")
        
        evaluator.save_results()
        evaluator.generate_comparison_table()
        evaluator.generate_visualization_data()

        print("\n" + "="*80)
        print("EWALUACJA ZAKOŃCZONA POMYŚLNIE")
        print("="*80)
        
        avg_score = sum(r['overall_quality_score'] for r in evaluator.evaluation_results) / len(evaluator.evaluation_results)
        
        print(f"\nFINALNE PODSUMOWANIE:")
        print(f"   • Oceniono raportów: {len(evaluator.evaluation_results)}")
        print(f"   • Średnia ocena: {avg_score:.2f}/100")
        print(f"   • Całkowity czas: {total_time:.2f}s")
        print(f"   • Średni czas/raport: {total_time/len(evaluator.evaluation_results):.2f}s")
        
        best = max(evaluator.evaluation_results, key=lambda x: x['overall_quality_score'])
        worst = min(evaluator.evaluation_results, key=lambda x: x['overall_quality_score'])
        
        print(f"\n Najlepszy raport: {best['stock_symbol']} ({best['overall_quality_score']:.2f}/100)")
        print(f" Najsłabszy raport: {worst['stock_symbol']} ({worst['overall_quality_score']:.2f}/100)")
        
        print(f"\n Wygenerowane pliki:")
        print(f"   • evaluation_results_auto.json - szczegółowe wyniki")
        print(f"   • evaluation_comparison_auto.md - tabela porównawcza")
        print(f"   • evaluation_viz_data.json - dane do wizualizacji")
        
    else:
        print("\n  Brak zebranych wyników")


if __name__ == "__main__":
    main()