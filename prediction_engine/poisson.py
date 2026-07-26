# prediction_engine/poisson.py
import math
import random
from prediction_engine.elo import expected_goals, expected_score

DC_RHO = -0.13

def dc_tau(a, b, lambda_, mu, rho=DC_RHO):
    if a == 0 and b == 0:
        return 1.0 - lambda_ * mu * rho
    if a == 0 and b == 1:
        return 1.0 + lambda_ * rho
    if a == 1 and b == 0:
        return 1.0 + mu * rho
    if a == 1 and b == 1:
        return 1.0 - rho
    return 1.0

def poisson_pmf(k, lambda_):
    if lambda_ <= 0.0:
        return 1.0 if k == 0 else 0.0
    try:
        p = math.exp(-lambda_)
        for i in range(1, k + 1):
            p *= lambda_ / i
        return p
    except OverflowError:
        return 0.0

def poisson_sample(lambda_, rng=None):
    if rng is None:
        rng = random.random
    L = math.exp(-lambda_)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng()
        if p <= L:
            break
    return k - 1

def match_prob(rating_a, rating_b, home_bonus_a=0.0):
    lambda_ = expected_goals(rating_a, rating_b, home_bonus_a)
    mu = expected_goals(rating_b, rating_a, -home_bonus_a / 2.0)
    
    win_a = 0.0
    draw = 0.0
    win_b = 0.0
    
    for a in range(9): # 0 to 8 goals
        p_a = poisson_pmf(a, lambda_)
        for b in range(9): # 0 to 8 goals
            tau = dc_tau(a, b, lambda_, mu, DC_RHO)
            p = p_a * poisson_pmf(b, mu) * tau
            
            if a > b:
                win_a += p
            elif a < b:
                win_b += p
            else:
                draw += p
                
    total = win_a + draw + win_b
    if total <= 0.0:
        return {"winA": 0.33, "draw": 0.33, "winB": 0.33, "expectedGoalsA": lambda_, "expectedGoalsB": mu}
        
    return {
        "winA": win_a / total,
        "draw": draw / total,
        "winB": win_b / total,
        "expectedGoalsA": lambda_,
        "expectedGoalsB": mu
    }

def sample_match(rating_a, rating_b, home_bonus_a=0.0, allow_draw=True, rng=None):
    if rng is None:
        rng = random.random
        
    e_a = expected_goals(rating_a, rating_b, home_bonus_a)
    e_b = expected_goals(rating_b, rating_a, -home_bonus_a / 2.0)
    
    goals_a = poisson_sample(e_a, rng)
    goals_b = poisson_sample(e_b, rng)
    
    if not allow_draw and goals_a == goals_b:
        # Knockout penalty shootout nudge based on Elo win expectancy
        if rng() < expected_score(rating_a, rating_b, home_bonus_a):
            goals_a += 1
        else:
            goals_b += 1
            
    return {"goalsA": goals_a, "goalsB": goals_b}
