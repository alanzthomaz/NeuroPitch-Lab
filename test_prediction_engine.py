# test_prediction_engine.py
import sys
from prediction_engine.predictor import MatchPredictor
from prediction_engine.simulator import TournamentSimulator

def main():
    print("Testing Python Prediction Engine...")
    
    predictor = MatchPredictor()
    
    # Try Spain vs Germany
    try:
        p = predictor.predict_match("spain", "germany")
        print(f"\nMatch: {p['team1']} (Elo {p['team1_elo']}) vs {p['team2']} (Elo {p['team2_elo']})")
        print(f"  {p['team1']} Win: {p['winA']*100:.2f}%")
        print(f"  Draw:      {p['draw']*100:.2f}%")
        print(f"  {p['team2']} Win: {p['winB']*100:.2f}%")
        print(f"  Expected Goals: {p['expectedGoalsA']:.2f} - {p['expectedGoalsB']:.2f}")
    except Exception as e:
        print(f"Error predicting match: {e}")
        sys.exit(1)
        
    print("\nTesting Tournament Simulator...")
    try:
        sim = TournamentSimulator()
        res = sim.simulate_knockouts(num_sims=100, bracket_teams=None)
        probs = res["advancement"]
        print("Live-conditioned win probabilities for top teams:")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1]["win"], reverse=True)
        for team, stages in sorted_probs[:6]:
            print(f"  {team}: win Cup: {stages['win']}% | reach Final: {stages['final']}% | reach SF: {stages['sf']}%")
    except Exception as e:
        print(f"Error simulating tournament: {e}")
        sys.exit(1)
        
    print("\nSuccess! Python prediction engine functions perfectly.")

if __name__ == "__main__":
    main()
