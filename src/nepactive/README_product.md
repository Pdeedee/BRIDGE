  两种求解策略：                                                                                                                               
  - lowest_energy：能量最低的 top 10 解                                                                                                        
  - most_molecules：分子数最多的 top 10 解                                               
                                                                                                                                               
  输出 YAML 文件 (solutions.yaml)，格式如：                                                                                                    
  composition:                                                                                                                                 
    C: 6
    H: 6
    O: 12
    N: 12
  lowest_energy:
    - molecules: {CO2: 6, H2O: 3, N2: 6}
      total_energy: -273.12
      molecule_count: 15
    - ...
  most_molecules:
    - molecules: {CO: 6, H2: 3, O2: 3, N2: 6}
      total_energy: -210.45
      molecule_count: 18
    - ...

  结构输出： 对每种策略的最优解（第1个），生成结构并保存优化前和优化后两个文件：
  - product_lowest_energy/before_opt_000.vasp
  - product_lowest_energy/after_opt_000.vasp
  - product_most_molecules/before_opt_000.vasp
  - product_most_molecules/after_opt_000.vasp

  用法：
  # 只求解，输出 yaml
  python make_product.py POSCAR --only-solve

  # 求解 + 每种方案生成3个结构
  python make_product.py POSCAR -n 3

  # 指定密度和输出前缀
  python make_product.py POSCAR -d 1600 -o CL20
