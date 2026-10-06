import argparse, json, sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--task',required=True)
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
a=p.parse_args(); sys.argv=[sys.argv[0]]
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True); app=launcher.app; env=None
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    cfg=parse_env_cfg(a.task,device='cuda:0',num_envs=100);cfg.seed=24
    agent=load_cfg_from_registry(a.task,'rsl_rl_cfg_entry_point');agent.seed=24
    env=gym.make(a.task,cfg=cfg);raw=env.unwrapped
    env=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device)
    runner.load(a.checkpoint);policy=runner.get_inference_policy(device=raw.device)
    obs=env.get_observations()
    original_compute=raw.termination_manager.compute
    trace=[]
    def capture_compute():
        done=original_compute()
        robot=raw.scene['robot']
        pos=robot.data.root_pos_w
        vel=robot.data.root_lin_vel_w
        ground=raw.scene.terrain.ground_height(pos)
        up=-robot.data.projected_gravity_b[:,2]
        trace.append(torch.cat((pos.clone(),vel.clone(),ground[:,None].clone(),up[:,None].clone(),done[:,None].clone(),raw.termination_manager.time_outs[:,None].clone(),raw.termination_manager.terminated[:,None].clone()),dim=1))
        return done
    raw.termination_manager.compute=capture_compute
    finished=torch.zeros(100,dtype=torch.bool,device=raw.device)
    totals=torch.zeros(100,dtype=torch.float64,device=raw.device)
    steps=torch.zeros(100,dtype=torch.long,device=raw.device)
    counts={'time_limit_without_termination':0,'low_clearance':0,'no_ground':0}
    with torch.inference_mode():
        for i in range(raw.max_episode_length):
            obs,rewards,dones,extras=env.step(policy(obs))
            active=~finished;totals[active]+=rewards[active];steps[active]+=1
            ended=active & dones.bool();s=trace[-1]
            counts['time_limit_without_termination']+=int((ended & raw.reset_time_outs & ~raw.reset_terminated).sum())
            counts['no_ground']+=int((ended & ~torch.isfinite(s[:,6])).sum())
            counts['low_clearance']+=int((ended & torch.isfinite(s[:,6]) & ((s[:,2]-s[:,6])<.31)).sum())
            finished |= dones.bool()
            if finished.all().item():break
    t=torch.stack(trace).cpu(); st=steps.cpu()
    active=torch.arange(t.shape[0])[:,None]<st[None,:]
    clear=t[:,:,2]-t[:,:,6]
    valid=active & torch.isfinite(clear)
    env0_steps=int(st[0]);env0=t[:env0_steps,0]
    result={'evaluation_version':'2026-10-05-restored-grid-average-friction-unfixed-initial-contact','task':a.task,'checkpoint':a.checkpoint,'seed':24,'num_envs':100,'completed':int(finished.sum()),'reward_mean':totals.mean().item(),'reward_std':totals.std(unbiased=False).item(),'steps_mean':steps.double().mean().item(),'steps_std':steps.double().std(unbiased=False).item(),'termination_counts':counts,'first_episodes':{'clearance_mean':clear[valid].mean().item(),'clearance_max':clear[valid].max().item(),'vertical_speed_abs_mean':t[:,:,5][active].abs().mean().item(),'vertical_speed_abs_max':t[:,:,5][active].abs().max().item(),'upright_mean':t[:,:,7][active].mean().item()},'env0':{'steps':env0_steps,'reward':totals[0].item(),'clearance_min':(env0[:,2]-env0[:,6]).min().item(),'clearance_max':(env0[:,2]-env0[:,6]).max().item(),'vertical_speed_min':env0[:,5].min().item(),'vertical_speed_max':env0[:,5].max().item(),'minimum_upright':env0[:,7].min().item(),'end_position':env0[-1,:3].tolist(),'reset_times_s':((t[:,0,8].nonzero().flatten()+1)*raw.step_dt).tolist(),'samples_half_sec':[{'time':round((j+1)*raw.step_dt,3),'clearance':float(t[j,0,2]-t[j,0,6]),'vertical_speed':float(t[j,0,5]),'upright':float(t[j,0,7])}for j in range(0,len(t),30)]}}
    Path(a.output).write_text(json.dumps(result,indent=2))
    print('BOUNCE_RESULT',json.dumps(result),flush=True)
finally:
    if env is not None:env.close()
    app.close()
