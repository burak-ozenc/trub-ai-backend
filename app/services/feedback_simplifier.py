from typing import Dict, Any
from app.core.models import SimpleFeedbackResult


class FeedbackSimplifier:
    """Convert complex analysis into simple, actionable feedback"""

    def simplify_analysis(self, technical_data: Dict[str, Any],
                          exercise_focus: str) -> SimpleFeedbackResult:
        """
        Convert technical analysis to simple feedback
        
        Args:
            technical_data: Complex analysis results
            exercise_focus: What the exercise is focusing on (breathing, tone, etc.)
        
        Returns:
            SimpleFeedbackResult with actionable tips
        """

        # Determine overall status
        overall_status = self._determine_status(technical_data, exercise_focus)

        # Identify main issue
        main_issue = self._identify_main_issue(technical_data, exercise_focus)

        # Generate quick tip
        quick_tip = self._generate_quick_tip(technical_data, exercise_focus, main_issue)

        # Suggest next step
        next_step = self._suggest_next_step(overall_status, main_issue, exercise_focus)

        return SimpleFeedbackResult(
            overall_status=overall_status,
            main_issue=main_issue,
            quick_tip=quick_tip,
            next_step=next_step
        )

    def _determine_status(self, data: Dict[str, Any], focus: str) -> str:
        """Determine overall performance status"""

        # Simple scoring based on focus area
        if focus == "breathing":
            consistency = data.get("breath_control", {}).get("breath_consistency", "")
            if "excellent" in consistency.lower() or "consistent" in consistency.lower():
                return "Excellent breathing! 🎺"
            elif "good" in consistency.lower():
                return "Good progress!"
            else:
                return "Let's improve your breathing"

        elif focus == "tone":
            quality = data.get("tone_quality", {}).get("quality_score", "")
            if "excellent" in quality.lower() or "rich" in quality.lower():
                return "Beautiful tone! 🎵"
            elif "good" in quality.lower():
                return "Nice tone quality!"
            else:
                return "Work on your tone"

        elif focus == "rhythm":
            consistency = data.get("rhythm_timing", {}).get("consistency", "")
            if "excellent" in consistency.lower() or "steady" in consistency.lower():
                return "Solid rhythm! 🎶"
            elif "good" in consistency.lower():
                return "Good timing!"
            else:
                return "Let's work on rhythm"

        # Default
        return "Keep practicing!"

    def _identify_main_issue(self, data: Dict[str, Any], focus: str) -> str:
        """Identify the primary issue to work on"""

        if focus == "breathing":
            breath_data = data.get("breath_control", {})
            consistency = breath_data.get("breath_consistency", "").lower()

            if "short" in consistency or "quick" in consistency:
                return "Your breaths are too short"
            elif "irregular" in consistency or "inconsistent" in consistency:
                return "Your breathing timing is uneven"
            elif "infrequent" in consistency:
                return "You're not breathing often enough"
            else:
                return None

        elif focus == "tone":
            tone_data = data.get("tone_quality", {})
            quality = tone_data.get("quality_score", "").lower()

            if "weak" in quality or "thin" in quality:
                return "Your tone needs more support"
            elif "harsh" in quality or "strained" in quality:
                return "Your embouchure is too tight"
            elif "airy" in quality or "breathy" in quality:
                return "Too much air escaping"
            else:
                return None

        elif focus == "rhythm":
            rhythm_data = data.get("rhythm_timing", {})
            consistency = rhythm_data.get("consistency", "").lower()

            if "rushing" in consistency:
                return "You're playing too fast"
            elif "dragging" in consistency:
                return "You're playing too slow"
            elif "unsteady" in consistency:
                return "Your tempo is wavering"
            else:
                return None

        return None

    def _generate_quick_tip(self, data: Dict[str, Any], focus: str, issue: str) -> str:
        """Generate one actionable tip"""

        tips_by_issue = {
            "Your breaths are too short": "Try: Breathe in for 4 counts through your nose, then play for 8 counts",
            "Your breathing timing is uneven": "Practice breathing every 4 measures at the same spot",
            "You're not breathing often enough": "Mark breath points in your music every 4-8 measures",
            "Your tone needs more support": "Use more air support from your diaphragm - imagine pushing from your stomach",
            "Your embouchure is too tight": "Relax your lips slightly and think 'open throat'",
            "Too much air escaping": "Firm up your corners and blow a focused air stream",
            "You're playing too fast": "Practice with a metronome at 75% speed first",
            "You're playing too slow": "Use a metronome and gradually increase by 5 BPM",
            "Your tempo is wavering": "Count out loud while playing to stay steady",
        }

        # If we have a specific tip for this issue, use it
        if issue and issue in tips_by_issue:
            return tips_by_issue[issue]

        # Otherwise, give a general tip based on focus
        general_tips = {
            "breathing": "Practice breathing exercises without the trumpet - 4 counts in, 4 counts out",
            "tone": "Long tones are your best friend - hold one note for 10 seconds",
            "rhythm": "Tap your foot steadily while playing to internalize the beat",
        }

        return general_tips.get(focus, "Keep practicing this exercise slowly and carefully")

    def _suggest_next_step(self, status: str, issue: str, focus: str) -> str:
        """Suggest what to do next"""

        if "excellent" in status.lower() or "beautiful" in status.lower() or "solid" in status.lower():
            return "Great job! Try this exercise at a faster tempo or move to the next difficulty level"

        elif issue:
            return f"Practice this exercise again, focusing on: {issue.lower()}"

        else:
            return f"Repeat this exercise a few more times to master the {focus} technique"