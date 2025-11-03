import numpy as np
from scipy.optimize import linprog, milp
import random
import time
from itertools import combinations

class AdvancedMolecularSolver:
    def __init__(self, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        else:
            current_seed = int(time.time() * 1000) % 10000
            random.seed(current_seed)
            np.random.seed(current_seed)
            print(f"使用随机种子: {current_seed}")
        
        # 定义分子及其原子组成 [C, H, O, N]
        self.molecules = {
            'CH4': [1, 4, 0, 0],
            'CO': [1, 0, 1, 0],
            'CO2': [1, 0, 2, 0],
            'H2': [0, 2, 0, 0],
            'H2O': [0, 2, 1, 0],
            'N2': [0, 0, 0, 2],
            'NH3': [0, 3, 0, 1],
            'O2': [0, 0, 2, 0],
            'CHN': [1, 1, 0, 1],
        }

        self.molecule_energies = {
            'CO': -14.747749,
            'NH3': -19.33091,
            'O2': -9.876768,
            'CHN': -19.782274,
            'CO2': -22.8497,
            'H2O': -14.086947,
            'CH4': -24.092106,
            'N2': -16.628838,
            'H2': -6.7345247,
        }
        
        # 创建原子-分子矩阵和相关数组
        self.molecule_names = list(self.molecules.keys())
        self.atom_matrix = np.array([self.molecules[name] for name in self.molecule_names]).T
        self.energy_array = np.array([self.molecule_energies[name] for name in self.molecule_names])

    def analyze_feasibility(self, c, h, o, n):
        """分析问题的可行性"""
        target_atoms = np.array([c, h, o, n])
        
        print(f"\n=== 可行性分析 ===")
        print(f"目标原子: C={c}, H={h}, O={o}, N={n}")
        
        # 分析原子比例
        ratios = {}
        total_atoms = c + h + o + n
        ratios['C'] = c / total_atoms
        ratios['H'] = h / total_atoms
        ratios['O'] = o / total_atoms
        ratios['N'] = n / total_atoms
        
        print(f"原子比例: C={ratios['C']:.3f}, H={ratios['H']:.3f}, O={ratios['O']:.3f}, N={ratios['N']:.3f}")
        
        # 检查每种分子单独能消耗的原子数量
        print(f"\n每种分子的理论最大数量:")
        for i, mol_name in enumerate(self.molecule_names):
            mol_atoms = self.atom_matrix[:, i]
            if np.any(mol_atoms > 0):
                max_count = min(target_atoms[j] // mol_atoms[j] for j in range(4) if mol_atoms[j] > 0)
                remaining = target_atoms - mol_atoms * max_count
                print(f"  {mol_name}: 最多{max_count}个, 剩余原子: C={remaining[0]}, H={remaining[1]}, O={remaining[2]}, N={remaining[3]}")
        
        # 使用线性规划检查可行性（放松整数约束）
        A_eq = self.atom_matrix
        b_eq = target_atoms
        bounds = [(0, None) for _ in range(len(self.molecules))]
        c_obj = np.ones(len(self.molecules))  # 任意目标函数
        
        try:
            result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if result.success:
                print(f"\n线性规划可行性检查: 可行")
                print(f"连续解: {[f'{mol}:{val:.2f}' for mol, val in zip(self.molecule_names, result.x) if val > 0.01]}")
                return True, result.x
            else:
                print(f"\n线性规划可行性检查: 不可行")
                print(f"失败原因: {result.message}")
                return False, None
        except Exception as e:
            print(f"可行性检查失败: {e}")
            return False, None

    def smart_rounding_solution(self, c, h, o, n):
        """智能四舍五入方法"""
        target_atoms = np.array([c, h, o, n])
        
        # 首先求解连续问题
        feasible, continuous_solution = self.analyze_feasibility(c, h, o, n)
        
        if not feasible:
            print("连续问题都不可行，尝试近似解")
            return self.approximate_solution(c, h, o, n)
        
        print(f"\n=== 智能四舍五入求解 ===")
        
        # 尝试多种四舍五入策略
        best_solution = None
        best_error = float('inf')
        
        strategies = [
            "floor",      # 向下取整
            "ceil",       # 向上取整  
            "round",      # 四舍五入
            "smart"       # 智能取整
        ]
        
        for strategy in strategies:
            print(f"\n尝试 {strategy} 策略:")
            
            if strategy == "floor":
                solution = np.floor(continuous_solution).astype(int)
            elif strategy == "ceil":
                solution = np.ceil(continuous_solution).astype(int)
            elif strategy == "round":
                solution = np.round(continuous_solution).astype(int)
            else:  # smart
                solution = self.smart_round(continuous_solution, target_atoms)
            
            # 计算误差
            calculated_atoms = np.dot(self.atom_matrix, solution)
            error = np.sum(np.abs(calculated_atoms - target_atoms))
            
            print(f"  解: {dict(zip(self.molecule_names, solution))}")
            print(f"  计算原子: C={calculated_atoms[0]}, H={calculated_atoms[1]}, O={calculated_atoms[2]}, N={calculated_atoms[3]}")
            print(f"  误差: {error}")
            
            if error < best_error:
                best_error = error
                best_solution = solution
        
        # 如果还没找到完美解，尝试局部搜索改进
        if best_error > 0:
            print(f"\n进行局部搜索改进...")
            best_solution, best_error = self.local_search_improvement(best_solution, target_atoms, max_iterations=100)
        
        return best_solution, best_error

    def smart_round(self, continuous_solution, target_atoms):
        """智能取整策略"""
        solution = np.zeros(len(continuous_solution), dtype=int)
        remaining_atoms = target_atoms.copy().astype(float)
        
        # 按连续解的大小排序，优先处理数量大的分子
        sorted_indices = np.argsort(-continuous_solution)
        
        for idx in sorted_indices:
            if continuous_solution[idx] < 0.01:
                continue
                
            mol_atoms = self.atom_matrix[:, idx].astype(float)
            
            # 计算在不违反约束的情况下可以取的最大整数
            if np.any(mol_atoms > 0):
                max_possible = min(remaining_atoms[i] / mol_atoms[i] for i in range(4) if mol_atoms[i] > 0)
                max_int = int(np.floor(max_possible))
                
                # 在连续解附近选择最合适的整数
                continuous_val = continuous_solution[idx]
                candidates = [max(0, int(np.floor(continuous_val))), max(0, int(np.ceil(continuous_val))), max_int]
                candidates = list(set([c for c in candidates if c <= max_int]))
                
                best_choice = 0
                best_residual = float('inf')
                
                for candidate in candidates:
                    test_remaining = remaining_atoms - mol_atoms * candidate
                    residual = np.sum(np.maximum(0, -test_remaining))  # 只考虑负残差
                    if residual < best_residual:
                        best_residual = residual
                        best_choice = candidate
                
                solution[idx] = best_choice
                remaining_atoms -= mol_atoms * best_choice
        
        return solution

    def local_search_improvement(self, initial_solution, target_atoms, max_iterations=50):
        """局部搜索改进"""
        current_solution = initial_solution.copy()
        current_atoms = np.dot(self.atom_matrix, current_solution)
        current_error = np.sum(np.abs(current_atoms - target_atoms))
        
        print(f"局部搜索开始，初始误差: {current_error}")
        
        for iteration in range(max_iterations):
            improved = False
            
            # 单分子调整
            for mol_idx in range(len(self.molecules)):
                for delta in [-2, -1, 1, 2]:
                    new_solution = current_solution.copy()
                    new_solution[mol_idx] = max(0, new_solution[mol_idx] + delta)
                    
                    new_atoms = np.dot(self.atom_matrix, new_solution)
                    new_error = np.sum(np.abs(new_atoms - target_atoms))
                    
                    if new_error < current_error:
                        current_solution = new_solution
                        current_atoms = new_atoms
                        current_error = new_error
                        improved = True
                        
                        if current_error == 0:
                            print(f"在迭代 {iteration+1} 找到完美解!")
                            return current_solution, current_error
            
            # 双分子同时调整
            if not improved:
                for i in range(len(self.molecules)):
                    for j in range(i+1, len(self.molecules)):
                        for delta_i in [-1, 1]:
                            for delta_j in [-1, 1]:
                                new_solution = current_solution.copy()
                                new_solution[i] = max(0, new_solution[i] + delta_i)
                                new_solution[j] = max(0, new_solution[j] + delta_j)
                                
                                new_atoms = np.dot(self.atom_matrix, new_solution)
                                new_error = np.sum(np.abs(new_atoms - target_atoms))
                                
                                if new_error < current_error:
                                    current_solution = new_solution
                                    current_atoms = new_atoms
                                    current_error = new_error
                                    improved = True
                                    
                                    if current_error == 0:
                                        print(f"在迭代 {iteration+1} 双重调整找到完美解!")
                                        return current_solution, current_error
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
                
            if iteration % 10 == 0:
                print(f"迭代 {iteration+1}: 误差 = {current_error}")
        
        print(f"局部搜索结束，最终误差: {current_error}")
        return current_solution, current_error

    def approximate_solution(self, c, h, o, n):
        """近似求解方法"""
        target_atoms = np.array([c, h, o, n])
        
        print(f"\n=== 近似求解 ===")
        
        # 构建松弛问题：允许一些原子有剩余
        num_atoms = 4
        num_molecules = len(self.molecules)
        
        # 目标函数：最小化原子使用的"不平衡度"
        c_obj = np.ones(num_molecules + num_atoms)  # 分子数量 + 原子剩余量
        c_obj[num_molecules:] = 100  # 惩罚原子剩余
        
        # 约束：使用的原子 + 剩余原子 >= 目标原子
        A_ub = np.zeros((num_atoms, num_molecules + num_atoms))
        A_ub[:, :num_molecules] = -self.atom_matrix  # 负号因为是 >= 约束转为 <= 约束
        A_ub[:, num_molecules:] = -np.eye(num_atoms)
        
        b_ub = -target_atoms
        
        bounds = [(0, None) for _ in range(num_molecules + num_atoms)]
        
        try:
            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success:
                solution = np.round(result.x[:num_molecules]).astype(int)
                
                calculated_atoms = np.dot(self.atom_matrix, solution)
                error = np.sum(np.abs(calculated_atoms - target_atoms))
                
                print(f"近似解: {dict(zip(self.molecule_names, solution))}")
                print(f"计算原子: C={calculated_atoms[0]}, H={calculated_atoms[1]}, O={calculated_atoms[2]}, N={calculated_atoms[3]}")
                print(f"目标原子: C={target_atoms[0]}, H={target_atoms[1]}, O={target_atoms[2]}, N={target_atoms[3]}")
                print(f"误差: {error}")
                
                return solution, error
            else:
                print("近似求解也失败了")
                return None, float('inf')
                
        except Exception as e:
            print(f"近似求解出错: {e}")
            return None, float('inf')

    def solve_difficult_case(self, c, h, o, n):
        """专门处理困难案例的求解方法"""
        print(f"\n=== 困难案例求解: C={c}, H={h}, O={o}, N={n} ===")
        
        # 1. 可行性分析
        feasible, continuous_solution = self.analyze_feasibility(c, h, o, n)
        
        # 2. 尝试智能四舍五入
        solution, error = self.smart_rounding_solution(c, h, o, n)
        
        if error == 0:
            print(f"\n找到完美解!")
            self.print_solution(solution, c, h, o, n)
            return solution, error
        
        print(f"\n未找到完美解，最佳误差: {error}")
        self.print_solution(solution, c, h, o, n)
        return solution, error

    def print_solution(self, solution, c, h, o, n):
        """打印解的详细信息"""
        target_atoms = np.array([c, h, o, n])
        calculated_atoms = np.dot(self.atom_matrix, solution)
        total_energy = np.dot(solution, self.energy_array)
        
        print(f"\n=== 解的详细信息 ===")
        solution_dict = {mol: count for mol, count in zip(self.molecule_names, solution) if count > 0}
        print(f"分子组合: {solution_dict}")
        print(f"使用原子: C={calculated_atoms[0]}, H={calculated_atoms[1]}, O={calculated_atoms[2]}, N={calculated_atoms[3]}")
        print(f"目标原子: C={target_atoms[0]}, H={target_atoms[1]}, O={target_atoms[2]}, N={target_atoms[3]}")
        print(f"原子差值: C={target_atoms[0]-calculated_atoms[0]}, H={target_atoms[1]-calculated_atoms[1]}, O={target_atoms[2]-calculated_atoms[2]}, N={target_atoms[3]-calculated_atoms[3]}")
        print(f"总误差: {np.sum(np.abs(calculated_atoms - target_atoms))}")
        print(f"总能量: {total_energy:.6f}")


# 测试困难案例
if __name__ == "__main__":
    solver = AdvancedMolecularSolver()
    
    # 你提到的困难案例
    c, h, o, n = 40, 64, 96, 32
    
    result_solution, result_error = solver.solve_difficult_case(c, h, o, n)