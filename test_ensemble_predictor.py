# test_ensemble_predictor.py
from ensemble.ensemble_predictor import EnsemblePredictor

def main():
    print("Testing Ensemble Predictor...")
    ens = EnsemblePredictor()
    p = ens.predict_match("spain", "germany")
    print("\nMatch: Spain vs Germany")
    print(f"  Stat Engine win Spain: {p['winA']*100:.2f}% | Draw: {p['draw']*100:.2f}% | win Germany: {p['winB']*100:.2f}%")
    print(f"  ANN Model   win Spain: {p['ann_prob_home']*100:.2f}% | Draw: {p['ann_prob_draw']*100:.2f}% | win Germany: {p['ann_prob_away']*100:.2f}%")
    print(f"  Ensemble    win Spain: {p['ensemble_home']*100:.2f}% | Draw: {p['ensemble_draw']*100:.2f}% | win Germany: {p['ensemble_away']*100:.2f}%")
    print("\nEnsemble predictor functions successfully!")

if __name__ == '__main__':
    main()
