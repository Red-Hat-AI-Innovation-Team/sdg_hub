/**
 * sdg-hub plugin for OpenCode.ai
 *
 * Registers skills directory and injects bootstrap context.
 * Tells the agent where to find sdg-hub scripts.
 */

import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const SDGHubPlugin = async ({ client, directory }) => {
  const pluginRoot = path.resolve(__dirname, '../..');
  const skillsDir = path.join(pluginRoot, 'skills');
  const scriptsDir = path.join(pluginRoot, 'scripts');

  const getBootstrapContent = () => {
    return `<sdg-hub-plugin>
You have the sdg-hub synthetic data generation plugin installed.

**Available skills:**
- data-generation — generate synthetic data using flows
- setup-guide — first-time configuration

**Script paths (use these instead of \${CLAUDE_PLUGIN_ROOT}):**
- Detection: ${scriptsDir}/sdg_detect.sh
- Generation: ${scriptsDir}/sdg_generate.sh
- Flows: ${scriptsDir}/sdg_flows.sh

When skills reference \${CLAUDE_PLUGIN_ROOT}/scripts/..., substitute the paths above.
</sdg-hub-plugin>`;
  };

  return {
    config: async (config) => {
      config.skills = config.skills || {};
      config.skills.paths = config.skills.paths || [];
      if (!config.skills.paths.includes(skillsDir)) {
        config.skills.paths.push(skillsDir);
      }
    },

    'experimental.chat.system.transform': async (_input, output) => {
      const bootstrap = getBootstrapContent();
      if (bootstrap) {
        (output.system ||= []).push(bootstrap);
      }
    }
  };
};
