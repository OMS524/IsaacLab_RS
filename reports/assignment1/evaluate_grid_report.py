"""Report-only first-episode evaluation, matching play_one_episode.py aggregation.

Writes independent report data/video; does not change tasks, checkpoints, or policy exports.
"""
import argparse, json, sys, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--task',required=True)
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--video-folder')
a=p.parse_args(); sys.argv=[sys.argv[0]]
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,enable_cameras=bool(a.video_folder))
app=launcher.app; env=None
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    cfg=parse_env_cfg(a.task,device='cuda:0',num_envs=100);cfg.seed=24
    cfg.viewer.origin_type='asset_root';cfg.viewer.asset_name='robot';cfg.viewer.env_index=0
    cfg.viewer.eye=(-4.0,4.0,2.5);cfg.viewer.lookat=(0.0,0.0,0.5)
    agent=load_cfg_from_registry(a.task,'rsl_rl_cfg_entry_point');agent.seed=24
    env=gym.make(a.task,cfg=cfg,render_mode='rgb_array' if a.video_folder else None)
    if a.video_folder:
        env=gym.wrappers.RecordVideo(env,video_folder=a.video_folder,step_trigger=lambda step:step==0,video_length=960,disable_logger=True)
    raw=env.unwrapped
    env=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device)
    runner.load(a.checkpoint);policy=runner.get_inference_policy(device=raw.device)
    obs=env.get_observations();initial=raw.scene['robot'].data.root_pos_w.clone()
    original_compute=raw.termination_manager.compute;trace=[]
    def capture_compute():
        done=original_compute();robot=raw.scene['robot']
        pos=robot.data.root_pos_w;vel=robot.data.root_lin_vel_w
        ground=raw.scene.terrain.ground_height(pos)
        trace.append(torch.cat((pos.clone(),vel.clone(),ground[:,None].clone(),(-robot.data.projected_gravity_b[:,2:3]).clone(),robot.data.root_lin_vel_b[:,1:2].clone(),robot.data.root_ang_vel_b[:,:2].clone()),dim=1))
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
            finished|=dones.bool()
            if (i+1)%240==0:print('REPORT_PROGRESS',i+1,'completed',int(finished.sum()),flush=True)
            if finished.all().item():break
    assert finished.all().item(),'Incomplete first episodes'
    t=torch.stack(trace).cpu();st=steps.cpu();active=torch.arange(t.shape[0])[:,None]<st[None,:]
    end=t[st-1,torch.arange(100),:3]
    result={'evaluation_version':'2026-10-05-grid06-final-three-model-report','task':a.task,'checkpoint':a.checkpoint,'seed':24,'num_envs':100,'video_folder':a.video_folder,'reward_terms':raw.reward_manager.active_terms,'completed':int(finished.sum()),'reward_mean':totals.mean().item(),'reward_std':totals.std(unbiased=False).item(),'steps_mean':steps.double().mean().item(),'steps_std':steps.double().std(unbiased=False).item(),'termination_counts':counts,'metrics':{'vertical_speed_abs_mean':float(t[:,:,5][active].abs().mean()),'lateral_speed_abs_mean':float(t[:,:,8][active].abs().mean()),'roll_pitch_rate_l2_mean':float(t[:,:,9:11].square().sum(-1)[active].mean()),'upright_mean':float(t[:,:,7][active].mean()),'x_displacement_mean':float((end[:,0]-initial.cpu()[:,0]).mean())},'episode_rewards':totals.cpu().tolist(),'episode_steps':st.tolist(),'max_steps':raw.max_episode_length,'step_dt':raw.step_dt}
    Path(a.output).write_text(json.dumps(result,indent=2))
    print('REPORT_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['episode_rewards','episode_steps']}),flush=True)
except BaseException:
    import traceback
    traceback.print_exc();print('REPORT_EVAL_FAIL',flush=True)
    raise
finally:
    if env is not None:env.close()
    app.close()
